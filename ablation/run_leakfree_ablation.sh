#!/usr/bin/env bash
# DGM Polyglot leak-free diagnose ablation — 2-arm runner.
#
# Arm A (leaky / published):   DGM_DIAGNOSE_ALLOW_TEST_LEAK=1 -> published DGM
#                              (diagnose LLM is shown the hidden grader tests).
# Arm B (leak-free / DEFAULT): no env (default)            -> diagnose LLM sees
#                              only solver-visible signal.
#
# Both arms share ONE initial agent + ONE initial eval (the leak is only in the
# self-improvement diagnose step, not initial solving), so any post-improvement
# delta between arms isolates the value of the test-leak to DGM's evolution.
#
# This is a LABELED ABLATION, not a replacement baseline. The canonical DGM
# baseline is Arm A. Do not report Arm B as "the DGM baseline".
#
# NOTE ON POWER: DGM's loop is stochastic and the gain is cumulative+noisy. A
# few generations at 1 repeat is illustrative only. For a materiality claim use
# REPEATS>=3 and a larger GENERATION_LIMIT, and compare archive-best across arms.
#
# Requirements: Docker, the baseline venv, and OPENAI_API_KEY (read from
# configs/providers/.env.openai if not already exported).
set -euo pipefail

DGM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KCSI_ROOT="$(cd "$DGM_DIR/../.." && pwd)"

# --- config (override via env) ----------------------------------------------
GENERATION_LIMIT="${GENERATION_LIMIT:-3}"
REPEATS="${REPEATS:-1}"
WALLCLOCK_CAP_SEC="${WALLCLOCK_CAP_SEC:-5400}"   # per outer-loop arm
BASELINE_VENV_DIR="${BASELINE_VENV_DIR:-$KCSI_ROOT/.venv_baselines}"
DATASET="${DGM_POLYGLOT_DATASET:-$KCSI_ROOT/benchmarks/polyglot/source/polyglot_benchmark_metadata.json}"
TASK_MAP="${DGM_POLYGLOT_TASK_MAP:-}"            # REQUIRED: a polyglot_N.json task-id map
OUT_ROOT="${OUT_ROOT:-$DGM_DIR/ablation/output_leakfree}"
DRY_RUN="${DRY_RUN:-false}"

# DGM model roles (default to the repo's OpenAI preset; o3-mini matches paper).
export DGM_CODE_MODEL="${DGM_CODE_MODEL:-gpt-5.4-mini}"
export DGM_SELF_IMPROVE_MODEL="${DGM_SELF_IMPROVE_MODEL:-gpt-5.4-mini}"
export DGM_OPENAI_MODEL="${DGM_OPENAI_MODEL:-gpt-5.4-mini}"
export DGM_DIAGNOSE_MODEL="${DGM_DIAGNOSE_MODEL:-gpt-5.4-mini}"
export DGM_REASONING_EFFORT="${DGM_REASONING_EFFORT:-medium}"

if [[ -z "$TASK_MAP" ]]; then
  echo "ERROR: set DGM_POLYGLOT_TASK_MAP to a polyglot task-id map (e.g. a polyglot_10.json)." >&2
  exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  prof="$KCSI_ROOT/configs/providers/.env.openai"
  if [[ -f "$prof" ]]; then
    OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' "$prof" | head -1 | cut -d= -f2-)"
    export OPENAI_API_KEY
  fi
fi
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY not set and not found in configs/providers/.env.openai" >&2; exit 2; }

mkdir -p "$OUT_ROOT"
echo "[ablation] dgm=$DGM_DIR gens=$GENERATION_LIMIT repeats=$REPEATS model=$DGM_DIAGNOSE_MODEL task_map=$TASK_MAP"

newest_output_dir() { ls -dt "$DGM_DIR"/output_dgm/*/ 2>/dev/null | head -1; }

run_arm() {
  local arm="$1" allow_leak_val="$2" rep="$3"
  local tag="${arm}_rep${rep}"
  local pred_dir="$OUT_ROOT/$tag/initial_eval/predictions"
  local report_dir="$OUT_ROOT/$tag/initial_eval/reports"
  local initial_dir="$OUT_ROOT/$tag/initial_polyglot"
  local model_name="initial_polyglot_${tag}"
  mkdir -p "$pred_dir" "$report_dir"

  echo "[ablation] === arm=$arm rep=$rep (DGM_DIAGNOSE_ALLOW_TEST_LEAK='${allow_leak_val}') ==="
  if [[ "$DRY_RUN" == "true" ]]; then echo "[dry-run] would run initial eval + outer loop for $tag"; return 0; fi

  # 1) initial eval (identical agent for both arms; leak is not here)
  ( cd "$DGM_DIR" && env "UV_PROJECT_ENVIRONMENT=$BASELINE_VENV_DIR" uv run \
      python polyglot/run_initial_eval.py \
      --dataset-path "$DATASET" --task-map "$TASK_MAP" \
      --pred-dir "$pred_dir" --output-dir "$report_dir" --model-name "$model_name" )
  local report_path; report_path="$(ls -t "$report_dir"/*.json 2>/dev/null | head -1)"
  [[ -n "$report_path" ]] || { echo "no initial-eval report for $tag" >&2; return 1; }
  ( cd "$DGM_DIR" && env "UV_PROJECT_ENVIRONMENT=$BASELINE_VENV_DIR" uv run \
      python polyglot/make_initial_metadata.py \
      --report "$report_path" --task-map "$TASK_MAP" \
      --output-dir "$initial_dir" --predictions-dir "$pred_dir" )

  # 2) outer self-improvement loop with the leak gate set per-arm
  ( cd "$DGM_DIR" && \
    DGM_DIAGNOSE_ALLOW_TEST_LEAK="$allow_leak_val" \
    timeout --signal=TERM --kill-after=60 "$WALLCLOCK_CAP_SEC" \
    env "UV_PROJECT_ENVIRONMENT=$BASELINE_VENV_DIR" uv run \
    python DGM_outer.py --polyglot --shallow_eval \
    --max_generation "$GENERATION_LIMIT" --selfimprove_size 1 --selfimprove_workers 1 \
    --num_swe_evals 1 \
    --polyglot_small_subset "$TASK_MAP" --polyglot_medium_subset "$TASK_MAP" \
    --polyglot_initial_dir "$initial_dir" --no_full_eval ) || \
    echo "[ablation] arm=$arm rep=$rep outer loop returned non-zero (timeout/err absorbed)"

  # record which output_dgm dir this arm produced
  local produced; produced="$(newest_output_dir)"
  echo "$produced" > "$OUT_ROOT/$tag/output_dgm_dir.txt"
  echo "[ablation] arm=$arm rep=$rep -> $produced"

  # reap stray eval/agent containers between arms
  docker ps -aq --filter "ancestor=dgm:latest" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 || true
}

for rep in $(seq 1 "$REPEATS"); do
  run_arm leaky    "1" "$rep"   # opt into the published leak
  run_arm leakfree ""  "$rep"   # default = leak-free
done

echo "[ablation] done. Outputs under $OUT_ROOT/. Compare archive-best resolved counts: leaky vs leakfree."
