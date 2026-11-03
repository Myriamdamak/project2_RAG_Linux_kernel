# Measured cost optimization — 3 November 2026

The generation-only change passes the requested target: **70.67% lower model
cost per request**, with **1.33 percentage points lower answer accuracy** on
the same 125-question evaluation. Overall accuracy, including correct refusals,
falls exactly two percentage points.

| Metric | Before: GPT-5.4 mini | After: GPT-5.4 nano | Change |
|---|---:|---:|---:|
| Mean generation + rewrite cost/request | $0.001580736 | $0.000463636 | **−70.67%** |
| Cost per 1,000 requests | $1.581 | $0.464 | −$1.117 |
| Answer accuracy, 113 answerable questions | 93.36% | 92.04% | **−1.33 points** |
| Overall accuracy including refusals, 125 questions | 94.00% | 92.00% | −2.00 points |
| Correct refusals, 12 unanswerable questions | 12/12 | 11/12 | −1 question |
| Generation p50 | 1,129.4 ms | 1,178.2 ms | +4.32% |
| Generation p95 | 2,160.1 ms | 2,135.9 ms | −1.12% |

**The cost criterion passes. A 50% end-to-end latency improvement has not been
established.** The candidate's full pipeline p95 is not measured by this paired
generation experiment.

## What changed

Generation now uses `openai:gpt-5.4-nano`, with `reasoning_effort=none`.
Rewriting and the accuracy judge continue to use GPT-5.4 mini. Both models have
explicit token prices in `librarian/config.py`; cached and uncached input and
reasoning output are accounted for separately.

The comparison retains the same BGE embeddings, Qdrant collection/generation,
HNSW ef=256, BM25 fusion, cross-encoder, rerank pool of 30, six evidence chunks,
IDK threshold, system prompt, and rewrite policy. Answer caching is disabled.
No index rebuild or upload is required.

## Measurement method

`eval/optimize_generation.py` runs paired comparisons against `data/qa.jsonl`.
For every question, it runs the baseline pipeline against the current index,
then gives nano exactly the same retrieved chunks in the same order. Both
answers are independently judged with GPT-5.4 mini and the same judge prompt
and gold reference. The existing reranker-based refusal gate is shared.

The final run contains 125 fully judged pairs: 113 answerable questions and
12 unanswerable questions. There are no missing judgments. Answer accuracy is
`(correct + 0.5 * partial) / 113`, matching `generation_eval.py`. Overall
accuracy uses the same answer scores plus whether each unanswerable question
was refused. The acceptance check enforces the two-point limit for both.

The cost column uses returned API token usage and the verified price table.
It includes each variant's generation cost plus the same measured rewrite
cost. Judge costs are recorded separately and excluded from serving cost.
Qdrant hosting and local embedding/reranking compute are excluded. Prompt-cache
hits are counted at the discounted rate; the savings are not inferred solely
from advertised prices.

The baseline pipeline is measured live. Candidate generation latency is
measured directly, but shared retrieval/rewrite stages are not rerun for nano.
Consequently, the paired runner cannot establish candidate end-to-end p95.
The earlier October generation report and the ten-request load test are
historical references, not the before column in this comparison.

## Evidence and reproduction

- `results/optimization_nano.json`: complete comparison, configuration, exact
  question manifest, index metadata, accuracy gate, costs, and generation latency.
- `results/optimization_nano.checkpoint.json`: saved answers, source excerpts,
  usage, and judgments for both variants; ignored by git.
- `eval/optimize_generation.py`: resumable paired evaluation and acceptance check.

Run from the project root in PowerShell:

```powershell
python -u eval/optimize_generation.py --use-dotenv-key --resume --workers 2
```

That command reuses this completed checkpoint. To conduct an independent fresh
comparison, use a new tag:

```powershell
python -u eval/optimize_generation.py --use-dotenv-key --workers 2 --tag optimization_nano_repeat
```

API generation and judge calls incur charges. `--limit 2` runs a pilot, whose
incomplete results cannot pass the acceptance check. Resume without `--limit`
to finish the identical question manifest. The credential-source option reads
the local `.env` key instead of an inherited shell key; no keys are stored in
the report. Different questions, prompts, index metadata, or retrieval settings
invalidate the checkpoint fingerprint.

## Serving and load testing

The validated generation model is set in `librarian/config.py`, `.env.example`,
and the local `.env`. Restart the running server to load it:

```powershell
python -m librarian.server
```

A separate load test can measure live end-to-end latency:

```powershell
python eval/load_test.py --url http://localhost:8000 --n 125 --concurrency 1 --tag nano_live
```

Keep the same question order, request count, concurrency, index, and cache policy
when comparing load runs. A matched live mini run would be needed to claim an
end-to-end p95 improvement. The cost reduction above does not depend on repeated
questions or prewarming the answer cache.

## Quality limitation

Nano answered one unanswerable question (`q117`) that mini correctly refused.
This regression is retained in the evaluation, rather than removed from the
scoring set. The accepted accuracy changes are measurements on this 125-question
set, not a guarantee about all future traffic. Review low-confidence answers and
rerun the full comparison when changing the corpus, model, or retrieval policy.

Pricing references, checked 2026-11-03:
[nano](https://developers.openai.com/api/docs/models/gpt-5.4-nano),
[mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

## Follow-up: common-answer caching

The matched nano cache benchmark is recorded separately in
`results/cache_nano.md` and `results/cache_nano.json`. Ten common questions were
asked five times in each arm, with an initially empty cache. At an 80% hit rate,
overall HTTP p95 improved 35.82% (7.40 s to 4.75 s), repeated-request p95 improved
99.45% (6.34 s to 35 ms), and model cost/request fell 78.78%.

The cache preserves answer/source/citation content. All forty live hits and a
separate replay of all 125 previously judged nano outputs passed identity checks.
This is not a new judged accuracy score. The cache workload and earlier model
comparison have different request distributions, so their savings are reported
separately. Cache misses still use the full nano pipeline.
