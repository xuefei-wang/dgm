"""Deterministic check that DGM_DIAGNOSE_LEAKFREE removes the test/answer leak.

Upstream DGM feeds the self-improvement *diagnose* LLM the benchmark's
solver-hidden grader content (Polyglot `reference_tests` / `reference_answers`,
SWE-bench `test_patch`). This test renders the real diagnose prompt with the
gate OFF and ON and asserts the secret test source is present only when OFF.

No API / Docker / keys needed. Run from the dgm repo root:
    python ablation/test_diagnose_leakfree.py
"""

import os
import sys

# dgm repo root so `prompts` and `utils` import cleanly regardless of cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import prompts.self_improvement_prompt as sip  # noqa: E402

SECRET_TESTS = "SECRET_TESTS_marker_7f3a"
SECRET_ANSWER = "SECRET_ANSWER_marker_91be"


def _render(leakfree: bool) -> str:
    """Render the polyglot diagnose user prompt with I/O stubbed out."""
    # Stub the filesystem-backed log discovery so we exercise only the gating.
    sip.find_selfimprove_eval_logs = lambda *a, **k: ([], [], [], [])
    sip.process_selfimprove_eval_logs = lambda *a, **k: ("MDLOG", "EVALLOG", "PREDPATCH", {})
    sip.get_current_code = lambda *a, **k: "CODE"
    # Force the deterministic main branch (skip the 25% stochastic branch).
    # The function does a function-local `import random`, so patch the stdlib module.
    import random as _random

    _random.random = lambda: 1.0

    dataset = [
        {
            "instance_id": "python__sentinel",
            "language": "python",
            "reference_answers": SECRET_ANSWER,
            "reference_tests": SECRET_TESTS,
            "problem_statement": "ISSUE TEXT",
        }
    ]

    if leakfree:
        os.environ["DGM_DIAGNOSE_LEAKFREE"] = "1"
    else:
        os.environ.pop("DGM_DIAGNOSE_LEAKFREE", None)

    _system, user = sip.get_diagnose_prompt_polyglot(
        "python__sentinel", "initial", _ROOT, "/tmp/unused_outdir", dataset
    )
    return user


def main() -> int:
    leaky = _render(leakfree=False)
    safe = _render(leakfree=True)

    failures = []
    # Baseline (gate off) must expose the hidden test source — proves the leak exists.
    if SECRET_TESTS not in leaky:
        failures.append("gate OFF: expected reference_tests in diagnose prompt, not found")
    # Leak-free (gate on) must NOT expose it, and should show the placeholder.
    if SECRET_TESTS in safe:
        failures.append("gate ON: reference_tests STILL present (leak not closed)")
    if SECRET_ANSWER in safe:
        failures.append("gate ON: reference_answers present (answer leak not closed)")
    if sip._LEAKFREE_PLACEHOLDER not in safe:
        failures.append("gate ON: leak-free placeholder missing")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS: DGM_DIAGNOSE_LEAKFREE withholds reference_tests/answers from the diagnose prompt")
    print(f"  gate OFF -> secret present ({len(leaky)} chars), gate ON -> secret absent ({len(safe)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
