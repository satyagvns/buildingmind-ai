"""All labeled outcomes pass; adversarial injection rates are reported honestly."""

from evals.run_tool_calling import evaluate_tools


def test_labeled_tool_evaluations_are_reproducible():
    first, second = evaluate_tools(), evaluate_tools()
    assert first["metrics"] == second["metrics"]
    assert len(first["cases"]) == 24
    assert all(case["expected_checks_pass"] for case in first["cases"])
    assert first["metrics"]["authorization_enforcement"]["value"] == 1
    assert first["metrics"]["evidence_provenance_completeness"]["value"] == 1
    assert first["metrics"]["unsupported_claim_detection"]["value"] == 1
    assert first["metrics"]["unnecessary_mock_tool_call_rate"]["value"] == 0
    assert first["metrics"]["tool_argument_validity"]["value"] < 1
    assert first["metrics"]["tool_execution_success_rate"]["value"] < 1
    assert all(
        case["executions"] <= 5 if "executions" in case else case["gateway_attempts"] <= 5
        for case in first["cases"]
    )
    assert all(case["rounds"] <= 2 for case in first["cases"])
