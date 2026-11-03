# Nano answer-cache benchmark — 3 November 2026

Same workload in both arms: ten common logged questions, five passes, 50 HTTP
requests per arm, concurrency one. Both arms started with an empty answer cache.
The candidate's ten cold fills are included in its overall latency and cost.
The existing port-8000 app was not used; a temporary loopback server ran the
same Flask API and complete nano pipeline.

| Metric | Nano, cache off | Nano, cache on | Reduction |
|---|---:|---:|---:|
| Overall p50 | 4,424.88 ms | 12.49 ms | 99.72% |
| Overall p95, including cold fills | 7,395.25 ms | 4,745.95 ms | **35.82%** |
| Repeated-request p95 | 6,341.24 ms | 35.17 ms | **99.45%** |
| Mean cost/request | $0.00041540 | $0.00008816 | **78.78%** |
| Answer-cache hits | 0/50 | 40/50 | 80% hit rate |

Candidate misses still took a median 4,478.36 ms and p95 5,781.21 ms. The
candidate hit median was 12.13 ms. Caching helps repeat requests; it does not
remove the retrieval, reranking, or generation work for a new question.

This run does **not** establish a 50% overall p95 reduction. Its 35.82% reduction
includes the 20% cold requests. The 99.45% p95 reduction applies to the forty
matched repeat requests only. Results depend on the repeat rate, question mix,
provider prompt caching, and network variation. This is a small workload
measurement, not an estimate of all future traffic.

## Quality and accounting checks

All forty live cache hits returned the same answer, sources, citations, and
refusal status as their initial cold fill. All hits reported zero new API
usage and zero incremental model cost. Separately, the 125 previously judged
nano outputs were replayed through the cache: all 125 retained identical
answer/source/citation content. That replay is an identity check, not a new
accuracy judgment.

The nano accuracy comparison in `optimization_nano.json` remains a separate
125-question experiment. Its cost baseline has a different question mix;
these percentages should not be multiplied to claim a combined saving.

## Implementation and operation

- Per-process exact-question LRU cache, 1,000 entries, one-hour TTL.
- Keys preserve case and include filters, retrieval mode, selected chunk count,
  generator, configuration, and index metadata.
- Reads and writes deep-copy responses so callers cannot mutate stored content.
- Identical concurrent misses share one request; failures release waiters and
  are not stored.
- Restarting a process clears its cache. Multiple workers maintain independent
  caches. Paraphrases are separate questions.

The local `.env` enables caching. Restart the application to use it.
`GET /api/health` shows the active cache policy; `GET /api/stats` includes the
observed hit rate. Those stats include historical requests; use matched
benchmark reports for before/after comparisons.

Reproduce this workload with a fresh benchmark run (API calls incur charges):

```powershell
python -u eval/cache_benchmark.py --use-dotenv-key
```

The runner selects common questions from the pre-benchmark request log and
falls back to QA questions when needed. `cache_nano.json` records the exact
question manifest, prior frequencies, request sequence, configuration,
per-request timing, content hashes, and summary. Benchmark traffic is logged
separately under `logs/cache_benchmark/`.
