import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from optimize_generation import compare, summarize
from librarian.config import cost_usd


def test_optimization_requires_complete_runs_and_accuracy_budget():
    before = {"complete": True, "mean_cost_usd": .002,
              "accuracy": .95, "overall_accuracy_with_idk": .95}
    after = {**before, "mean_cost_usd": .001, "accuracy": .93,
             "overall_accuracy_with_idk": .93}
    assert compare(before, after)["target_met"]
    assert not compare(before, {**after, "accuracy": .925})["target_met"]
    assert not compare(before, {**after, "mean_cost_usd": .0011})["target_met"]
    assert not compare(before, {**after, "complete": False})["target_met"]


def test_summary_does_not_treat_missing_judgments_as_success():
    result = {"answerable": True, "idk": False, "timings_ms": {"generate": 100},
              "cost_usd": .001, "generation_cost_usd": .0009,
              "judge": {"correctness": "partial"}}
    report = summarize([{"before": result}], "before", 2)
    assert not report["complete"] and report["accuracy"] == .5
    assert report["overall_accuracy_with_idk"] == .5
    missing = summarize([], "after", 2)
    assert missing["accuracy"] is None and not missing["complete"]


def test_nano_usage_includes_cached_and_reasoning_tokens():
    cost = cost_usd("gpt-5.4-nano", {
        "input_tokens": 1000, "cached_input_tokens": 2000,
        "output_tokens": 100, "thinking_tokens": 100,
    })
    assert cost == pytest.approx((1000 * .2 + 2000 * .02 + 200 * 1.25) / 1e6)
