# DGM Polyglot — leak-free diagnose ablation

## What this is

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

## The gate

`prompts/self_improvement_prompt.py` adds an env gate, **default off**:

```
DGM_DIAGNOSE_LEAKFREE=1   # withhold reference_tests / reference_answers / test_patch
                          # from the diagnose prompt (replaced by a placeholder)
```

Unset (default) reproduces published DGM behavior exactly, so canonical baseline
numbers stay reproducible. This is a **labeled ablation, not a replacement
baseline** — the canonical DGM baseline is the leaky arm.

## Verify the gate (no API/Docker)

```
python ablation/test_diagnose_leakfree.py
# PASS: gate OFF -> hidden tests present in prompt; gate ON -> absent (+ placeholder)
```

## Run the 2-arm ablation (needs Docker + OPENAI_API_KEY)

```
DGM_POLYGLOT_TASK_MAP=/path/to/polyglot_10.json \
GENERATION_LIMIT=3 REPEATS=1 \
bash ablation/run_leakfree_ablation.sh
```

Both arms share one initial agent + one initial eval (leak is only in the
diagnose step), then run the outer loop with the gate off (`leaky`) vs on
(`leakfree`). Compare archive-best resolved counts across arms.

**Power caveat:** DGM's loop is stochastic and the gain is cumulative/noisy. A
few generations at `REPEATS=1` is illustrative only. For a materiality claim use
`REPEATS>=3` and a larger `GENERATION_LIMIT`.
