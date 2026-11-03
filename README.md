# The Librarian — ask the Linux kernel documentation

A retrieval-augmented question-answering service over the Linux kernel
`Documentation/` tree (release **v7.3**, ~4,600 documents, ~4 M words). Every
answer cites the passages it came from, and it says **"I don't know"** when
the documentation doesn't cover the question.

CS496 AI Engineering — Project 2.

### Structure-aware chunks and Qdrant

Chunking defaults to `structure`: it keeps section breadcrumbs and packs
paragraphs, code blocks, and tables within each section. Oversized blocks are
split at line or whitespace boundaries to respect the configured size.
`section` remains an alias, and `fixed` is available for baseline evaluations.

To store chunks and embeddings in Qdrant, set these values in your local `.env`:

```dotenv
CLUSTER_KEY=your-api-key
LIB_QDRANT_URL=https://your-cluster.region.cloud.qdrant.io:6333
LIB_QDRANT_COLLECTION=kernel_documentation
```

Then run `python -m librarian.build_index --chunker structure`. This rebuilds
the corpus, uploads vectors with full chunk metadata, and enables Qdrant dense
retrieval. BM25 and citation metadata remain local. The API key is never saved
in index metadata. Each build has its own generation filter, so older uploads
cannot appear in new search results. Old generations are retained in Qdrant;
remove them separately when they are no longer needed.

To preserve an existing index during migration, build into a separate directory:
`python -m librarian.build_index --out index_qdrant --chunker structure --reuse-index index`.
This reuses vectors only when both the embedding model and complete chunk text
match. After verifying the upload, set `LIB_INDEX_DIR=index_qdrant` in `.env`.
For a local copy of the configured model, set `LIB_EMBED_MODEL_PATH` to its directory.

Qdrant HNSW settings are explicit for new collections. To tune the existing
collection in place, run `python -m librarian.tune_qdrant --apply`. This keeps
the uploaded vectors and allows Qdrant to index segments in the background.
`LIB_QDRANT_HNSW_EF=256` controls query search effort; increasing it can improve
recall at the cost of latency. The query planner can still scan small filtered
sets when that is cheaper. Search requests return only IDs; chunk text is
already available locally.

Run `python -m librarian.tune_qdrant --benchmark` to compare approximate and
exact search using cached document vectors. It reports recall@50 and p50/p95
HTTP latency, excluding BGE query encoding, reranking, and answer generation.
Treat it as a database smoke benchmark; use representative question vectors
and the retrieval evaluation before drawing conclusions about overall quality.
Use `--benchmark --samples 24 --ef-values 64 128 256` to compare search efforts.

```
question ─► (rewrite if short/multi-part) ─► BM25 ┐
                                              vector ┴► RRF fusion ─► cross-encoder rerank ─► top 6
                                                                                     │
                     "I don't know" ◄── best score below threshold ◄─────────────────┤
                                                                                     ▼
                                        gpt-5.4-nano: answer from excerpts only, [n] citations
                                                                                     ▼
                                                     citations validated in code ─► answer + sources
```

## Setup and run

Requirements: Python 3.10+, git, ~3 GB disk, an OpenAI API key
([OpenAI Platform](https://platform.openai.com/api-keys)).

```bash
cp .env.example .env          # put your OPENAI_API_KEY in it
make setup                    # venv + deps + corpus (Documentation/ of v7.3) + index   (~15 min on a laptop CPU)
make run                      # http://localhost:8000
```


**Deployed:** `<add your URL here>` — e.g. a Hugging Face Space (Docker SDK,
`PORT=7860`) or Render.

### OpenAI configuration

For an existing checkout, add `OPENAI_API_KEY=your-key` to `.env` (do not replace
it with `.env.example` if it already contains settings). Restart the server after
changing environment variables. A Gemini key is no longer used.

```dotenv
OPENAI_API_KEY=your-key
LIB_GENERATOR=openai:gpt-5.4-nano
LIB_REWRITE_MODEL=gpt-5.4-mini
LIB_REASONING_EFFORT=none
```

Answers, query rewriting, and the evaluation judge use the OpenAI Responses API.
`LIB_OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`.
BGE embeddings, the Qdrant HNSW index, local BM25, and the cross-encoder remain
in use. Generation defaults to nano; query rewriting stays on mini. See
`optimisation.md` for the measured comparison. No index rebuild is needed.

### No key / no internet?
Everything has an offline mode, used for tests:
```bash
LIB_EMBEDDER=lsa LIB_RERANKER=none LIB_GENERATOR=extractive make index run
```

## API

```bash
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question": "What does vm.swappiness control?", "filters": {"subsystem": "admin-guide"}}'
```
Filters: `subsystem`, `doc_type` (`rst`|`txt`|`abi`), `path_prefix`, `updated_after` (`YYYY-MM-DD`).
Also `POST /api/feedback`, `GET /api/stats` (live p50/p95, cost, IDK rate, feedback), `GET /api/health`.

## Repository

| path | what |
|---|---|
| `librarian/corpus.py` | load rst / txt / sysfs ABI / kernel-parameters, keep metadata (title, date, author, section, version) |
| `librarian/chunker.py` | section-aware chunker (+ fixed-window baseline) |
| `librarian/bm25.py`, `embed.py`, `index.py`, `qdrant.py` | keyword index, BGE embeddings, Qdrant HNSW vector search |
| `librarian/retrieve.py`, `rerank.py` | hybrid search, RRF, metadata filters, cross-encoder |
| `librarian/generate.py`, `pipeline.py` | OpenAI generation, citation validation, IDK gate, rewrite, cache |
| `librarian/server.py` + `static/` | web UI and JSON API, request & feedback logs |
| `data/qa.jsonl` | 125 eval questions: 110 single-source, 3 multi-source, 12 unanswerable, each with verbatim source passage |
| `eval/` | `validate_qa.py`, `retrieval_eval.py` (recall@k + ablation), `chunk_sweep.py`, `generation_eval.py` (faithfulness, correctness, citation accuracy, IDK), `load_test.py` (p50/p95, cost model) |
| `design.md` | chunking, index choice, known failure cases |
| `optimisation.md` | Week 8 before/after |
| `postmortem.md` | what went wrong |

## Evaluation

```bash
make eval            # validate gold labels, retrieval ablation, generation eval with LLM judge
make sweep           # chunk size / overlap / strategy sweep
make run & make load # end-to-end latency + cost model
```

Retrieval and generation are measured separately: a correct answer from the
wrong passage counts as a retrieval bug.

The paired 125-question generation optimization reduced mean generation plus
rewrite cost from $0.001581 to $0.000464 per request (**70.67%**), with answer
accuracy changing from 93.36% to 92.04% (**−1.33 percentage points**). Generation
now uses nano. See [the measured optimization report](optimisation.md) and
[the comparison artifact](results/optimization_nano.json) for the configuration,
refusal regression, and methodology. Candidate end-to-end p95 was not measured.

To reproduce or resume the same comparison:

```powershell
python -u eval/optimize_generation.py --use-dotenv-key --resume --workers 2
```

### Results

Offline embedder (LSA), no reranker — **placeholder until the default
pipeline is run**; numbers below are real but from the weakest configuration:

| mode | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| BM25 | 0.637 | 0.788 | 0.876 | 0.729 |
| vector (LSA) | 0.212 | 0.522 | 0.602 | 0.359 |
| hybrid (RRF) | 0.451 | 0.717 | 0.779 | 0.580 |
| hybrid + rerank | *run with default config* | | | |

Generation, correctness, faithfulness, latency and cost: fill from
`results/generation_baseline.json` and `results/load_baseline.json`.

## Cost model

`gpt-5.4-nano` generation pricing: $0.20 / 1M input tokens, $0.02 / 1M
cached input tokens, and $1.25 / 1M output tokens
([official nano documentation](https://developers.openai.com/api/docs/models/gpt-5.4-nano),
checked 2026-11-03). Rewriting and the evaluation judge use `gpt-5.4-mini`:
$0.75 input, $0.075 cached input, and $4.50 output per million tokens
([official mini documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini)).
Query rewriting and generation are metered separately.
`eval/load_test.py` measures usage and projects traffic costs. Custom models
need their own verified entry in `librarian/config.py` for accurate estimates.
Embedding and reranking run locally on CPU with no API fee.

### Caching common answers

The local configuration enables an exact-question answer cache:

```dotenv
LIB_ANSWER_CACHE=true
LIB_ANSWER_CACHE_TTL=3600
LIB_ANSWER_CACHE_SIZE=1000
```

Each process keeps up to 1,000 answers for one hour using LRU eviction. Filters,
model/configuration, retrieval mode, and index metadata are part of the key;
capitalization is preserved. Identical concurrent requests share one generation.
Hits reuse the answer and citations with zero new API usage. Restart the app
after changing settings; restarting also clears the process's cache.

The nano cache benchmark used the same ten common questions asked five times
in each arm, including the initially empty cache. With an 80% hit rate, repeated
request p95 fell from 6.34 seconds to 35 ms, and overall p95 fell 35.82%, from
7.40 to 4.75 seconds. Cost/request fell 78.78%. These are repeated-workload
results; new questions still take the full pipeline path.

See [the cache benchmark report](results/cache_nano.md) and
[raw measurements](results/cache_nano.json). Re-run it with:

```powershell
python -u eval/cache_benchmark.py --use-dotenv-key
```

`GET /api/health` reports the active cache policy. `GET /api/stats` and the
regular load-test report show cache hit rates; the load test also splits hit
and miss latency.

## Contributions

| member | contributions |
|---|---|
| | |
| | |
| | |
| | |

## Licence and data
Code: MIT. Corpus: the Linux kernel documentation (GPL-2.0 / CC-BY-4.0),
fetched at setup, not redistributed in this repo. No API keys or personal data
are committed (`.env` is git-ignored; request logs stay on the server).

### Resume generation evaluation

Generation evaluation saves each answer before judging it and checkpoints completed
judgments. Temporary HTTP 429/500/502/503/504 responses receive bounded retries.
If a judge still fails, the evaluation continues and marks the report incomplete.
Resume with the same settings and tag to retry failures without regenerating saved answers:

```bash
python eval/generation_eval.py --judge --tag baseline --resume
```

Check `summary.complete` and `summary.judge_errors` before reporting metrics.
Checkpoints include source excerpts and usage; printed terminal progress alone cannot
restore missing answers, token costs, or claim-level judgments from an older run.
