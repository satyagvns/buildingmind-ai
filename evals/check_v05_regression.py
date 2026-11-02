"""Run the released v0.5 tests against released/current code; no branch changes."""

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.check_legacy_regression import SNAPSHOT

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "a580df1b7667f4340f9f3d2df00ad6e30e320498"
ADVANCED = """
import json
from examples.advanced_floor7 import create_pipeline
from examples.floor7_hvac import QUESTION
answer,report=create_pipeline().answer(QUESTION)
result={"answer":answer.model_dump(mode="json"),"retrieval":report.model_dump(mode="json")}
result["retrieval"].pop("elapsed_seconds")
result["retrieval"].pop("rerank_seconds")
print(json.dumps(result,sort_keys=True))
"""


def check_v05():
    with TemporaryDirectory(prefix="buildingmind-v05-") as folder:
        baseline = Path(folder)
        archive = subprocess.check_output(["git", "archive", BASELINE], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(baseline, filter="data")
        tests, snapshots = {}, {}
        for label, source, extra in (
            ("released_v05", baseline, []),
            ("current_source", ROOT, ["-k", "not test_package_version"]),
        ):
            env = {**os.environ, "PYTHONPATH": str(source / "src") + os.pathsep + str(baseline)}
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *extra],
                cwd=baseline,
                env=env,
                capture_output=True,
                text=True,
            )
            if run.returncode:
                raise RuntimeError(
                    f"v0.5 compatibility failed: {label}\n{run.stdout}\n{run.stderr}"
                )
            tests[label] = run.stdout.strip().splitlines()[-1]
            snapshots[label] = {
                name: json.loads(
                    subprocess.check_output(
                        [sys.executable, "-c", script], cwd=baseline, env=env, text=True
                    )
                )
                for name, script in (("legacy", SNAPSHOT), ("advanced", ADVANCED))
            }
        if snapshots["released_v05"] != snapshots["current_source"]:
            raise RuntimeError("Released v0.5 diagnostic responses or rankings changed")
        return {
            "baseline_commit": BASELINE,
            "test_results": tests,
            "intentional_exclusion": (
                "Only the archived package-version assertion is excluded against current source."
            ),
            "snapshot_comparison": (
                "equal: four legacy scenarios with normalized evidence IDs; "
                "full advanced response and ranking with timing fields removed"
            ),
            "compatibility": (
                "No existing evidence models, source IDs, retrieval/generation "
                "interfaces or diagnostic behavior changed."
            ),
        }


if __name__ == "__main__":
    print(json.dumps(check_v05(), indent=2))
