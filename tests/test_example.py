import json
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "floor7_hvac.py"


def test_offline_example_outputs_structured_answer():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=True
    )
    answer = json.loads(result.stdout)
    assert answer["status"] == "answered"
    assert len(answer["observations"]) == 7
    assert result.stderr == ""


def test_optional_openai_example_requires_model():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--provider", "openai"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--model is required" in result.stderr
