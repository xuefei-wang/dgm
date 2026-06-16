# DGM Polyglot — leak-free diagnose (default fix + ablation)

## The bug

By **upstream DGM design**, the self-improvement *diagnose* step is shown the
benchmark's solver-hidden grader content and its output reshapes the coding
agent used on future graded tasks:

- `prompts/self_improvement_prompt.py` → `get_diagnose_prompt_polyglot`
  sets `test_patch = entry["reference_tests"]` (the hidden Polyglot grader test
  source) and `answer_patch = entry["reference_answers"]` (the example solution),
  then interpolates `{test_patch}` into the diagnoser prompt under a
  `# Private Test Patch` section. (`get_diagnose_prompt_swe` does the same with
  SWE-bench's `test_patch`.)

This is an **information-parity leak**: a meta-LLM sees grader content a fair
solver never gets, during the loop that produces DGM's reported improvement
(Polyglot 14.2% → 30.7% in the paper). It fires only in the evolutionary loop
(`DGM_outer.py`), **not** in a plain initial eval — so the initial 14.2% is clean.

## The fix (default behavior in this fork)

`prompts/self_improvement_prompt.py` now **withholds the hidden grader content
from the diagnose prompt BY DEFAULT** (replaced by a placeholder). To reproduce
the published DGM behavior (e.g. to match the paper's numbers), opt back in:

```
DGM_DIAGNOSE_ALLOW_TEST_LEAK=1   # restore published DGM (diagnoser sees the private tests)
```

So the default is leak-free; the leak is available only as an explicit, labeled
opt-out.

## Verify (no API/Docker)

```
python ablation/test_diagnose_leakfree.py
# PASS: default -> hidden tests absent from prompt (+ placeholder);
#       DGM_DIAGNOSE_ALLOW_TEST_LEAK=1 -> hidden tests present
```

## A/B the two behaviors (needs Docker + OPENAI_API_KEY)

```
DGM_POLYGLOT_TASK_MAP=/path/to/polyglot_10.json \
GENERATION_LIMIT=3 REPEATS=1 \
bash ablation/run_leakfree_ablation.sh
```

Runs both arms with one shared initial agent + initial eval (the leak is only in
the diagnose step): `leaky` (`DGM_DIAGNOSE_ALLOW_TEST_LEAK=1`) vs `leakfree`
(default). Compare archive-best resolved counts across arms.

**Power / infra caveat:** DGM's loop is stochastic and cumulative; a few
generations at `REPEATS=1` is illustrative only. The polyglot eval defaults to
`--max-workers 5` — on a memory-constrained/shared host the concurrent rust/cpp
eval containers can OOM (exit 137), which dominates small runs. For a materiality
claim, cap eval concurrency (`--max-workers 1-2`) and use `REPEATS>=3`.
