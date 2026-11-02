"""Bounded fan-out. Backend adapters must honor timeout_seconds for network I/O."""

import math
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from time import perf_counter

from buildingmind.retrieval.fusion import reciprocal_rank_fusion
from buildingmind.retrieval.models import RetrievalReport


class HybridRetriever:
    def __init__(
        self,
        *,
        keyword,
        vector,
        top_k: int = 12,
        rrf_k: float = 60,
        max_workers: int = 2,
        timeout_seconds: float = 5,
        concurrent: bool = True,
        weights=None,
    ):
        if (
            top_k <= 0
            or not 1 <= max_workers <= 8
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Invalid retrieval limits")
        self.keyword, self.vector = keyword, vector
        self.top_k, self.rrf_k = top_k, rrf_k
        self.max_workers, self.timeout_seconds = max_workers, timeout_seconds
        reciprocal_rank_fusion([], k=rrf_k, weights=weights)
        self.concurrent, self.weights = concurrent, weights
        self._active = Lock()

    def retrieve(self, query, filters, *, mode: str = "hybrid") -> RetrievalReport:
        if mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        started = perf_counter()
        jobs = []
        for backend, texts in (
            ("keyword", query.keyword_queries),
            ("vector", query.semantic_queries),
        ):
            if mode not in {backend, "hybrid"}:
                continue
            for i, text in enumerate(texts):
                jobs.append((backend, i, text))
        if len(jobs) > 16:
            raise ValueError("Too many retrieval queries")

        def search(job):
            backend, _, text = job
            retriever = self.keyword if backend == "keyword" else self.vector
            options = {"min_score": 0} if backend == "vector" else {}
            hits = retriever.search(text, filters=filters, top_k=self.top_k, **options)
            if any(not filters.matches(h.evidence) for h in hits):
                raise ValueError("Backend returned unauthorized evidence")
            return hits

        if not self._active.acquire(blocking=False):
            return RetrievalReport(
                complete=False, errors=("retrieval busy: earlier backend work remains",)
            )
        if not jobs:
            self._active.release()
            return RetrievalReport(complete=False, errors=("No retrieval queries supplied",))
        pool = ThreadPoolExecutor(max_workers=self.max_workers if self.concurrent else 1)
        futures = [(job, pool.submit(search, job)) for job in jobs]
        remaining = len(futures)
        completion_lock = Lock()

        def finished(future):
            nonlocal remaining
            with completion_lock:
                remaining -= 1
                if remaining == 0:
                    self._active.release()

        for _, future in futures:
            future.add_done_callback(finished)
        done, _ = wait([f for _, f in futures], timeout=self.timeout_seconds)
        lists, errors = [], []
        for (backend, i, _), future in futures:
            if future not in done:
                future.cancel()
                errors.append(f"{backend}[{i}]: timeout")
                continue
            try:
                lists.append((backend, i, future.result()))
            except Exception:
                errors.append(f"{backend}[{i}]: retrieval failed")
        # Python cannot kill running threads; network adapters need their own I/O deadlines.
        pool.shutdown(wait=False, cancel_futures=True)
        return RetrievalReport(
            candidates=reciprocal_rank_fusion(lists, k=self.rrf_k, weights=self.weights),
            errors=tuple(errors),
            complete=not errors,
            elapsed_seconds=perf_counter() - started,
        )
