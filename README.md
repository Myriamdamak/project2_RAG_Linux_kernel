# The Librarian — ask the Linux kernel documentation

A retrieval-augmented question-answering service over the Linux kernel
`Documentation/` tree (release **v7.3**, ~4,600 documents, ~4 M words). Every
answer cites the passages it came from, and it says **"I don't know"** when
the documentation doesn't cover the question.

CS496 AI Engineering — Project 2.

```
question ─► (rewrite if short/multi-part) ─► BM25 ┐
                                              vector ┴► RRF fusion ─► cross-encoder rerank ─► top 6
                                                                                     │
                     "I don't know" ◄── best score below threshold ◄─────────────────┤
                                                                                     ▼
                                        gpt-5.4-mini: answer from excerpts only, [n] citations
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
LIB_GENERATOR=openai:gpt-5.4-mini
LIB_REWRITE_MODEL=gpt-5.4-mini
LIB_REASONING_EFFORT=none
```

Answers, query rewriting, and the evaluation judge use the OpenAI Responses API.
`LIB_OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`.
The current local BGE embeddings, BM25 index, and cross-encoder remain in use;
no index rebuild is needed for this provider change. Existing evaluation results
are historical and must be rerun to assess OpenAI answer quality and cost.

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
| `librarian/bm25.py`, `embed.py`, `index.py` | keyword index, embeddings, flat-file vector store |
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

`gpt-5.4-mini` standard pricing: $0.75 / 1M input tokens, $0.075 / 1M
cached input tokens, and $4.50 / 1M output tokens, including reasoning tokens
([official model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini),
checked 2026-10-27). Query rewriting and generation are metered separately.
`eval/load_test.py` measures usage and projects traffic costs. Custom models
need their own verified entry in `librarian/config.py` for accurate estimates.
Embedding and reranking run locally on CPU with no API fee.

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
