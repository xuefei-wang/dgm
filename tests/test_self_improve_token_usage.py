"""Tests for DGM self-improve (archive-evolution) token accounting (kcsi #1196)."""

import json

import DGM_outer


_META_LINE = (
    'TOKEN_USAGE {"cached_tokens": 5, "cost_usd": 0.5, "input_tokens": 100, '
    '"output_tokens": 40, "reasoning_tokens": 7, "total_tokens": 140}\n'
)
_SOLVE_LINE = (
    'TOKEN_USAGE {"cached_tokens": 0, "cost_usd": 9.9, "input_tokens": 9000, '
    '"output_tokens": 9000, "reasoning_tokens": 0, "total_tokens": 18000}\n'
)


def _make_run_tree(root):
    step = root / "20260707_000000_000000"
    (step / "predictions" / "model_0").mkdir(parents=True)
    # Meta-loop transcripts (should be counted).
    (step / "self_evo.md").write_text("# self evo\n" + _META_LINE, encoding="utf-8")
    (step / "self_improve.log").write_text("INFO diagnose\n" + _META_LINE, encoding="utf-8")
    # Per-task solve transcript under predictions/ (must NOT be counted).
    (step / "predictions" / "model_0" / "task.md").write_text(_SOLVE_LINE, encoding="utf-8")


def test_aggregate_counts_meta_loop_excludes_predictions(tmp_path):
    _make_run_tree(tmp_path)
    result = DGM_outer.aggregate_self_improve_token_usage(str(tmp_path))
    # Two meta-loop TOKEN_USAGE records; the predictions/ solve line is excluded.
    assert result["calls"] == 2
    assert result["prompt_tokens"] == 200
    assert result["completion_tokens"] == 80
    assert result["total_tokens"] == 280
    assert result["cached_prompt_tokens"] == 10
    assert result["uncached_prompt_tokens"] == 190
    assert result["reasoning_tokens"] == 14
    assert result["cost_usd"] == 1.0
    assert result["malformed_records"] == 0


def test_aggregate_writes_run_level_json(tmp_path):
    _make_run_tree(tmp_path)
    DGM_outer.aggregate_self_improve_token_usage(str(tmp_path))
    out_path = tmp_path / "self_improve_llm_usage.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["calls"] == 2
    assert written["total_tokens"] == 280


def test_aggregate_empty_tree_has_null_cost(tmp_path):
    result = DGM_outer.aggregate_self_improve_token_usage(str(tmp_path))
    assert result["calls"] == 0
    assert result["cost_usd"] is None


def test_aggregate_malformed_record_counted(tmp_path):
    step = tmp_path / "step"
    step.mkdir()
    (step / "self_evo.md").write_text("TOKEN_USAGE not-json\n" + _META_LINE, encoding="utf-8")
    result = DGM_outer.aggregate_self_improve_token_usage(str(tmp_path))
    assert result["malformed_records"] == 1
    assert result["calls"] == 1
