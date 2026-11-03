import sys
from pathlib import Path
from unittest.mock import Mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import load_test as evaluation


def test_http_502_preserves_underlying_error(monkeypatch):
    response = Mock(ok=False, status_code=502)
    response.json.return_value = {
        "error": "The librarian failed to answer.",
        "detail": "OpenAI API HTTP 429 (type=insufficient_quota, code=credit_balance_exhausted)",
    }
    monkeypatch.setattr(evaluation.requests, "post", Mock(return_value=response))
    result = evaluation.one("question", "http://localhost:8000")
    assert not result["ok"] and result["status"] == 502
    assert "credit_balance_exhausted" in result["error"]
    summary = evaluation.summarize([result], 1, 1, 200)
    assert summary["errors"] == 1 and not summary["complete"]
    assert summary["p50_ms"] is None and summary["p95_ms"] is None
    assert summary["cost_per_request_usd"] is None
    assert summary["cost_model"]["today"]["usd_per_month"] is None


def test_timeout_is_recorded(monkeypatch):
    monkeypatch.setattr(evaluation.requests, "post", Mock(side_effect=requests.Timeout("Request timed out")))
    result = evaluation.one("question", "http://localhost:8000")
    assert not result["ok"] and result["status"] is None
    assert "timed out" in result["error"]


def test_partial_failure_metrics_use_only_successful_requests():
    success = {"ok": True, "ms": 100, "cost": 0.001,
               "tokens": 100, "out_tokens": 20, "server": {"search": 10}}
    failure = {"ok": False, "status": 502, "ms": 500, "error": "upstream failure"}
    summary = evaluation.summarize([success, failure], 2, 1, 200)
    assert summary["p50_ms"] == 100 and summary["p95_ms"] == 100
    assert summary["cost_per_request_usd"] == 0.001
    assert summary["successful_requests"] == 1 and summary["errors"] == 1
    assert not summary["complete"]


def test_all_failures_save_report_and_return_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["load_test.py", "--n", "1", "--tag", "failure"])
    monkeypatch.setattr(evaluation, "load_qa", lambda: [{"question": "question"}])
    monkeypatch.setattr(evaluation, "one", lambda *args: {
        "ok": False, "status": 502, "ms": 1, "error": "upstream error"})
    save = Mock(return_value=tmp_path / "load_failure.json")
    monkeypatch.setattr(evaluation, "save", save)
    assert evaluation.main() == 1
    assert save.call_args.args[0] == "load_failure.json"
    assert save.call_args.args[1]["error_details"][0]["status"] == 502


def test_load_summary_separates_hit_and_miss_latency():
    miss = {"ok": True, "cached": False, "ms": 1000, "cost": .001,
            "tokens": 100, "out_tokens": 20, "server": {"generate": 900}}
    hit = {"ok": True, "cached": True, "ms": 5, "cost": 0,
           "tokens": 0, "out_tokens": 0, "server": {}}
    summary = evaluation.summarize([miss, hit], 2, 1, 200)
    assert summary["cache_hit_rate"] == .5 and summary["cache_hits"] == 1
    assert summary["cache_latency_ms"]["hit"]["p95"] == 5
    assert summary["cache_latency_ms"]["miss"]["p95"] == 1000
    assert summary["mean_input_tokens"] == 50
    assert summary["cost_per_request_usd"] == .0005
