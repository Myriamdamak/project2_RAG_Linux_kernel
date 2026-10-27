# Design — The Librarian

A RAG service over the Linux kernel `Documentation/` tree (release **v7.3**).
This file records each design decision, the evidence for it, and the failure
cases we know about. Numbers marked *(LSA)* come from the offline embedder;
re-run `make eval` with the default embedder for the report.

## 1. Corpus

| | |
|---|---|
| Source | `torvalds/linux`, tag `v7.3`, `Documentation/` only (sparse checkout) |
| Licence | GPL-2.0 / CC-BY-4.0 documentation, public. No personal data beyond author names in headers. |
| Size | 4,569 documents (3,170 `.rst`, 702 `.txt`, 697 ABI files), ~4 M words, ~41.5k chunks |
| Excluded | `translations/` (non-English duplicates), Sphinx build files, images, DT-binding YAML schemas |
| Users | Kernel developers, sysadmins, embedded engineers, and OS-course students (our classmates) |

Why it passes the five tests: it is the live documentation of the kernel
(real); people read it daily to tune sysctls, write drivers and submit patches
(real users); it is public; 4 M words cannot fit in a prompt (needs search);
and every fact sits in a specific file we can quote (checkable — see
`data/qa.jsonl`).

Pinning a tag matters: answers are only "correct" relative to one release.
Example from this very version: `vm.swappiness` now ranges **0–200**, while
most model training data (and most blog posts) still say 0–100.

## 2. Offline: building the index

### 2.1 Loading and metadata

`librarian/corpus.py` parses three source shapes differently:

| Shape | Parser | Section unit |
|---|---|---|
| `.rst` / `.txt` | heading detection by RST adornment rules; level = order of first appearance of each style | heading path `Title > Section > Subsection` |
| `ABI/*` (sysfs) | `What:` / `Date:` / `Description:` blocks | one sysfs file (`What:`) per section |
| `admin-guide/kernel-parameters.txt` | dedicated parser | one boot parameter per section |

Metadata kept per chunk: **title**, **section** path, **author(s)** (from
`:Author:` fields and "Copyright (c)" lines in the first 80 lines), **date**,
kernel **version**, **subsystem** (top-level directory), **doc_type**,
GitHub **url**.

**Date.** The fetch script keeps commit history since 2021-01-01 (blobless,
so no old file contents) and records each file's last-change date. Files not
touched since then get `<2021-01-01` instead of a fake date. The boundary
commits of a shallow clone "add" every file, so the first version of this
script dated ~2,400 files to January 2021; we now detect boundary commits
from `.git/shallow`.

Removed before indexing: `toctree`, `image`, `include`, SPDX headers, label
targets, substitutions, and `kernel-doc::` directives (see failure cases).

### 2.2 Chunking — tested, not assumed

`section` chunker (default): never crosses a section boundary; splits a
section into *blocks* (paragraph, whole literal/code block, whole table,
whole indented list item) and packs blocks greedily up to `chunk_tokens`,
with whole-block overlap up to `chunk_overlap`. A paragraph ending in `::`
stays glued to the code block it introduces. Each chunk is indexed with a
breadcrumb header `[path] Title > Section`, so a chunk that only says
"The default value is 60." still matches "swappiness default".

`fixed` chunker: sliding word windows over the whole file. Kept as a baseline.

The sweep (`eval/chunk_sweep.py`, results in `results/chunk_sweep.md`)
varies strategy, size (200 / 350 / 500 / 800) and overlap (0 / 50 / 100).
Because larger chunks contain the evidence span more often *by
construction*, the table also reports the prompt tokens that top-5 costs.
Our choice is the configuration with the best recall per prompt token, not
the best raw recall. → See §5 for the numbers.

### 2.3 Storage: flat files, not a vector database

| Option | Verdict |
|---|---|
| **Flat files (chosen)** — `chunks.jsonl`, `vectors.npy` (float16), BM25 as a scipy sparse matrix | 41.5k × 384 × 2 bytes = 32 MB. Exact cosine search is one matrix–vector product (~5 ms CPU). No server, reproducible, ships inside the Docker image. |
| Postgres + pgvector | Worth it for concurrent writes, joins, or >1M vectors. We have a read-only index rebuilt per kernel release. |
| Vector DB (Qdrant, Chroma…) | Approximate search buys nothing at 41k vectors and adds a service to deploy and pay for. |

Revisit at 100× *corpus* size (4M chunks ≈ 3 GB float16): then use an ANN
index (FAISS HNSW / pgvector HNSW). 100× *traffic* does not change this —
the index is read-only and replicates with the container.

### 2.4 Keyword index

BM25 (k1 = 1.2, b = 0.75) implemented in `librarian/bm25.py`. The tokenizer
is the important part for this corpus: it emits both the whole identifier and
its parts, so `CONFIG_PREEMPT_RT` matches queries for `CONFIG_PREEMPT_RT`,
`PREEMPT_RT` and "preempt rt". Only plurals are stemmed, because aggressive
stemming merges identifiers that must stay distinct.

### 2.5 Embeddings

Default `BAAI/bge-small-en-v1.5` (384-d, 33 M params, CPU-friendly, uses a
query instruction prefix). Alternative `lsa` (TF-IDF + SVD) needs no download
and is used for offline tests and as a baseline.

## 3. Online: answering a question

1. **Rewrite (conditional).** Only for questions of ≤ 3 words, with several
   `?`, comparisons ("vs", "difference between"), or long multi-part
   questions. `gemini-3.5-flash-lite` returns up to 3 search queries; all are
   searched and fused. Other questions skip the extra LLM call.
2. **Retrieve.** BM25 top-50 and vector top-50 per query, with optional
   metadata filters (`subsystem`, `doc_type`, `path_prefix`, `updated_after`)
   applied as masks before ranking.
3. **Fuse** with Reciprocal Rank Fusion (k = 60): uses ranks only, so BM25
   scores and cosines never need to share a scale.
4. **Rerank** the top 30 with the cross-encoder
   `cross-encoder/ms-marco-MiniLM-L-6-v2`; keep the top 6.
5. **"I don't know" gate.** If the best rerank score is below
   `idk_threshold`, answer "I don't know" without calling the LLM. The
   threshold is tuned on the unanswerable questions by
   `eval/retrieval_eval.py` so that at most 5% of answerable questions are
   refused.
6. **Generate** with `gemini-3.5-flash-lite` (temperature 0,
   `thinking_level = minimal`): numbered excerpts only, a citation after every
   factual sentence, exactly "I don't know" when the excerpts don't answer.
7. **Validate citations in code.** Citation numbers that don't exist are
   stripped and logged; citation coverage is computed per answer.

## 5. Measured results so far (offline configuration)

Embedder `lsa`, no reranker, 113 answerable questions. These are the floor:
the default embedder and the cross-encoder are expected to move vector,
hybrid and especially R@1 a lot. Re-run with `make eval` and `make sweep`.

### Retrieval ablation
Index: 41554 chunks, embedder `lsa`, chunker `section` 350/50 tokens. 113 answerable questions.

| mode | R@1 | R@3 | R@5 | R@10 | R@20 | MRR | doc@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.637 | 0.752 | 0.788 | 0.876 | 0.929 | 0.729 | 0.947 | 0.4 | 0.6 |
| vector | 0.212 | 0.407 | 0.522 | 0.602 | 0.699 | 0.359 | 0.823 | 40.2 | 58.2 |
| hybrid | 0.451 | 0.637 | 0.717 | 0.779 | 0.903 | 0.580 | 0.903 | 39.6 | 53.3 |

### Chunking sweep
| config | chunks | mean tok | ~tokens in top-5 | mode | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| fixed-350-50-lsa | 19938 | 389 | 1946 | bm25 | 0.557 | 0.850 | 0.911 | 0.706 |
| fixed-350-50-lsa | 19938 | 389 | 1946 | hybrid | 0.451 | 0.752 | 0.894 | 0.599 |
| section-200-0-lsa | 57905 | 125 | 626 | bm25 | 0.593 | 0.796 | 0.885 | 0.701 |
| section-200-0-lsa | 57905 | 125 | 626 | hybrid | 0.407 | 0.690 | 0.779 | 0.537 |
| section-200-50-lsa | 58744 | 128 | 639 | bm25 | 0.611 | 0.796 | 0.894 | 0.711 |
| section-200-50-lsa | 58744 | 128 | 639 | hybrid | 0.451 | 0.673 | 0.788 | 0.561 |
| section-350-0-lsa | 41364 | 176 | 878 | bm25 | 0.620 | 0.796 | 0.876 | 0.721 |
| section-350-0-lsa | 41364 | 176 | 878 | hybrid | 0.460 | 0.664 | 0.761 | 0.568 |
| section-350-50-lsa | 41554 | 178 | 888 | bm25 | 0.637 | 0.788 | 0.876 | 0.729 |
| section-350-50-lsa | 41554 | 178 | 888 | hybrid | 0.451 | 0.717 | 0.779 | 0.580 |
| section-350-100-lsa | 42144 | 183 | 914 | bm25 | 0.620 | 0.796 | 0.894 | 0.727 |
| section-350-100-lsa | 42144 | 183 | 914 | hybrid | 0.496 | 0.681 | 0.814 | 0.601 |
| section-500-50-lsa | 35744 | 205 | 1026 | bm25 | 0.611 | 0.823 | 0.885 | 0.723 |
| section-500-50-lsa | 35744 | 205 | 1026 | hybrid | 0.443 | 0.717 | 0.770 | 0.567 |
| section-800-100-lsa | 31877 | 231 | 1156 | bm25 | 0.575 | 0.832 | 0.903 | 0.712 |
| section-800-100-lsa | 31877 | 231 | 1156 | hybrid | 0.460 | 0.681 | 0.788 | 0.571 |

Reading the sweep:
- **Fixed windows win raw R@5 (0.85) but cost 2.2× the prompt tokens** of
  section-350 for the same top-5, because their chunks are twice as long. Per
  token spent, section-aware chunking is better; fixed windows also cut
  sysfs entries and boot parameters in half (F2).
- **Section 200 matches section 350 on R@5 at 70% of the tokens**, but has 40%
  more chunks to embed and a worse hybrid score. A real candidate for Week 8.
- **Overlap barely matters** inside sections (±0.02), because chunks rarely
  split a paragraph; 50 tokens is kept as a cheap safety margin.
- Provisional choice: **section, 350 tokens, 50 overlap** (best MRR). Confirm
  with the default embedder before the report.

### Generation (offline extractive baseline, no LLM)
`results/generation_extractive_lsa.json`: cites a gold passage in 54% of
answers, gold passage retrieved in top-6 for 74%, invalid citations 0% (2.5%
before the fix for F6). Gemini numbers come from `eval/generation_eval.py --judge`.

## 4. Known failure cases

| # | Case | Effect | Status |
|---|---|---|---|
| F1 | **`kernel-doc::` directives.** API references (e.g. `kmalloc()` parameters) are generated from C comments at build time; that text is not in `Documentation/`. | Function-level API questions get "I don't know" or a partial answer from prose docs. | Known, out of scope (would need parsing `include/` and `mm/` C sources). |
| F2 | **Long list files.** `kernel-parameters.txt` is 8,700 lines; generic chunking split entries mid-way and lost the parameter name. | Boot-parameter questions retrieved fragments without their name. | **Fixed** with a dedicated parser (one entry per section). Other long lists (e.g. `sysctl/*.rst`) work because each knob is a heading. |
| F3 | **Tables.** Grid/simple tables larger than a chunk are split by line; rows lose their column headers. | Answers from the middle of a big table (e.g. `process/changes.rst` versions) can misread columns. | Partly mitigated (tables kept whole when they fit). |
| F4 | **Code blocks.** A long listing becomes a chunk of code with no prose. | Retrieved but unhelpful; the model may describe code it doesn't understand. | `::` paragraphs stay attached to their code; long listings still split. |
| F5 | **Cross-references.** "See Documentation/x.rst" or `:ref:` links are not followed. | Answers stop at "see X" when the facts are in X. | Known. Fix idea: expand top hits with their referenced section. |
| F6 | **Bracketed numbers in the docs** (`fstests[2]`, footnotes) look like citations. | Bogus or mis-attributed citations (seen in 2.5% of baseline answers). | **Fixed**: escaped to `(2)` before the excerpts go to the model. |
| F7 | **Duplicated / obsolete content.** cgroup v1 vs v2, `ABI/obsolete` vs `ABI/testing`. | Correctly cited but outdated answer. | Header shows the path (`ABI/obsolete/...`) to the model; users can filter by subsystem. |
| F8 | **Version drift vs model prior.** e.g. swappiness 0–200 in v7.3. | Model "corrects" the docs with stale knowledge. | Prompt forbids outside knowledge; caught by the faithfulness judge. |
| F9 | **Non-English questions.** Our classmates often ask in French or Arabic; `bge-small-en` and BM25 are English-only, and translations are excluded. | Low recall on non-English questions. | Known. Option: `intfloat/multilingual-e5-small`, or rewrite to English first. |
| F10 | **Date metadata is shallow.** History kept since 2021 only. | Older files show `<2021-01-01`. | By design (the full history is ~4 GB). |
| F12 | **Rare identifiers vs RRF.** "What does `nosmp` do?": BM25 ranks the `nosmp` entry #1, the embedder doesn't find it at all, and RRF ranks chunks that *both* retrievers place ~#5 above it. | Without the reranker the right entry falls out of the top 5. | The reranker re-scores the top 30, where the entry still is. `LIB_BM25_WEIGHT` tests leaning on BM25 (with LSA: 1.5 → MRR 0.58→0.61, small). |
| F11 | **Eval-set bias.** We wrote the questions while reading the passages, which favours word overlap (BM25). | Recall is optimistic; BM25 looks better than it will on real users' questions. | Real user questions from the feedback log are added to the eval set as they arrive. |
