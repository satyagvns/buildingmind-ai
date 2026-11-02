"""Exercise archived v0.4 tests against both v0.4 and current source without a checkout."""

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "22cbddc"
SNAPSHOT = """
import json
from examples.floor7_hvac import create_pipeline, QUESTION
responses={}
for case in ("full","missing-telemetry","missing-records","missing-history"):
    p=create_pipeline()
    if case == "missing-telemetry": p.telemetry.evidence=()
    if case == "missing-records": p.retriever._entries.clear()
    if case == "missing-history": p.historical.evidence=()
    answer=p.answer(QUESTION).model_dump(mode="json")
    # Metadata defaults change revision hashes; compare all public fields after
    # replacing evidence IDs with stable source/location references.
    ids={c["evidence_id"]: f'{c["source_id"]}:{c["start_char"]}:{c["end_char"]}'
         for c in answer["citations"]}
    def normalize(value):
        if isinstance(value,dict): return {k:normalize(v) for k,v in value.items()}
        if isinstance(value,list): return [normalize(v) for v in value]
        if isinstance(value,str): return ids.get(value,value)
        return value
    responses[case]=normalize(answer)
print(json.dumps(responses,sort_keys=True))
"""


def check_legacy():
    with TemporaryDirectory(prefix="buildingmind-v04-") as folder:
        archived = Path(folder)
        archive = subprocess.check_output(["git", "archive", BASELINE], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(archived, filter="data")
        results, snapshots = {}, {}
        for label, source, excluded in (
            ("baseline", archived, []),
            ("current", ROOT, ["-k", "not test_package_version"]),
        ):
            env = {**os.environ, "PYTHONPATH": str(source / "src") + os.pathsep + str(archived)}
            test = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *excluded],
                cwd=archived,
                env=env,
                capture_output=True,
                text=True,
            )
            if test.returncode:
                raise RuntimeError(
                    f"{label} legacy regression failed:\n{test.stdout}\n{test.stderr}"
                )
            results[label] = test.stdout.strip().splitlines()[-1]
            snapshots[label] = json.loads(
                subprocess.check_output(
                    [sys.executable, "-c", SNAPSHOT], cwd=archived, env=env, text=True
                )
            )
        if snapshots["baseline"] != snapshots["current"]:
            raise RuntimeError("Legacy response snapshots changed")
        return {
            "baseline_commit": BASELINE,
            "test_results": results,
            "intentional_exclusion": (
                "Current source version is 0.5.0; exclude the archived 0.4.0 "
                "version assertion only for the current-source run."
            ),
            "snapshot_comparison": (
                "equal after normalizing revision-specific evidence IDs to source/offset references"
            ),
            "scenarios": list(snapshots["baseline"]),
            "normalized_output": snapshots["current"],
        }


if __name__ == "__main__":
    print(json.dumps(check_legacy(), indent=2))
