from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest

from librarian.config import settings
from librarian.pipeline import Librarian


@pytest.fixture
def library(monkeypatch):
    monkeypatch.setattr(settings, "answer_cache", True)
    monkeypatch.setattr(settings, "answer_cache_size", 2)
    monkeypatch.setattr(settings, "answer_cache_ttl", 3600.)
    lib = Librarian(SimpleNamespace(index=SimpleNamespace(meta={"generation": "first"})),
                    SimpleNamespace(model="gpt-5.4-nano"))
    calls = []

    def fresh(question, *args):
        calls.append(question)
        return {"answer": question, "sources": [{"text": "original"}],
                "cached": False, "cost_usd": .001, "usage": [{"input_tokens": 10}],
                "timings_ms": {"total": 100.}}

    monkeypatch.setattr(lib, "_ask_uncached", fresh)
    return lib, calls


def test_hit_has_no_new_usage_and_copies_nested_content(library):
    lib, calls = library
    original = lib.ask("question")
    original["sources"][0]["text"] = "caller mutation"
    hit = lib.ask("question")
    assert hit["cached"] and hit["cost_usd"] == 0 and hit["usage"] == []
    assert hit["sources"][0]["text"] == "original"
    hit["sources"][0]["text"] = "another mutation"
    assert lib.ask("question")["sources"][0]["text"] == "original"
    assert calls == ["question"]


def test_expiry_and_lru_eviction(library, monkeypatch):
    lib, calls = library
    clock = [0.]
    monkeypatch.setattr("librarian.pipeline.time.monotonic", lambda: clock[0])
    lib.ask("one")
    lib.ask("two")
    assert lib.ask("one")["cached"]
    lib.ask("three")  # evicts two, not the recently read one
    assert lib.ask("one")["cached"]
    assert not lib.ask("two")["cached"]
    clock[0] = 3601.
    assert not lib.ask("two")["cached"]
    assert calls.count("two") == 3


def test_keys_preserve_case_filters_model_and_index(library, monkeypatch):
    lib, _ = library
    lib.ask("What is %p?", {"subsystem": "core"})
    assert not lib.ask("What is %P?", {"subsystem": "core"})["cached"]
    assert not lib.ask("What is %p?", {"subsystem": "other"})["cached"]
    lib.ask("same")
    lib.generator.model = "gpt-5.4-mini"
    assert not lib.ask("same")["cached"]
    lib.retriever.index.meta["generation"] = "second"
    assert not lib.ask("same")["cached"]
    monkeypatch.setattr(settings, "top_k", settings.top_k + 1)
    assert not lib.ask("same")["cached"]


def test_concurrent_misses_share_one_generation(library, monkeypatch):
    lib, calls = library
    fresh = lib._ask_uncached
    started, release = threading.Event(), threading.Event()

    def slow(*args):
        started.set()
        assert release.wait(3)
        return fresh(*args)

    monkeypatch.setattr(lib, "_ask_uncached", slow)
    with ThreadPoolExecutor(4) as executor:
        futures = [executor.submit(lib.ask, "same") for _ in range(4)]
        assert started.wait(3)
        release.set()
        results = [future.result(timeout=3) for future in futures]
    assert calls == ["same"]
    assert sum(result["cached"] for result in results) == 3


def test_failures_do_not_cache_or_leave_waiters_blocked(library, monkeypatch):
    lib, _ = library
    fresh = lib._ask_uncached
    monkeypatch.setattr(lib, "_ask_uncached", lambda *a: (_ for _ in ()).throw(RuntimeError("upstream failure")))
    with pytest.raises(RuntimeError):
        lib.ask("same")
    assert not lib._inflight and not lib._cache
    monkeypatch.setattr(lib, "_ask_uncached", fresh)
    assert not lib.ask("same")["cached"]
