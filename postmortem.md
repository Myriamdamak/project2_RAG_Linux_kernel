# Postmortem

Problems we hit, what they cost, and what we changed. Kept honest on purpose:
the next team (or the next kernel release) will hit the same things.

> Started during the Week 5–7 build. Add Week 8 and deployment entries as they
> happen; each entry: what happened, impact, root cause, fix, lesson.

## 1. Half of the most useful file was unreadable to retrieval
**What:** `admin-guide/kernel-parameters.txt` (8,700 lines, every boot
parameter) was chunked like any other file. Chunks started mid-entry, e.g.
"…resets. * rstonce: only attempt one reset…", with no parameter name.
**Impact:** boot-parameter questions — among the most common ones — could
only succeed by luck. Our first eval set had *no* such question, so no metric
showed it; we found it by reading chunks.
**Fix:** a dedicated parser, one entry per section headed by the parameter
name; added boot-parameter questions to the eval set.
**Lesson:** look at the chunks of the files users care most about before
trusting any aggregate number.

## 2. The minimum chunk size silently deleted answers
**What:** `min_chunk_words = 25` was meant to drop toctree-only sections. It
also dropped short complete answers such as `quiet [KNL,EARLY] Disable most
log messages` (6 words).
**Fix:** lowered to 5 words; the validator now checks that every gold
evidence span survives chunking (`eval/validate_qa.py`).
**Lesson:** a filter tuned for noise needs a test that it doesn't remove signal.

## 3. Dates were wrong for ~2,400 files
**What:** the shallow clone's boundary commits look like they "add" every
file, so files untouched since 2021 were dated January 2021.
**Fix:** read the boundary commits from `.git/shallow` and mark those files
`<2021-01-01`.
**Lesson:** metadata derived from a partial history must say it is partial.

## 4. The docs' own `[2]` looked like citations
**What:** kernel docs contain footnotes like `fstests[2]`. Our citation
parser treated them as citations to source 2. 2.5% of baseline answers had a
bogus citation.
**Fix:** bracketed numbers in excerpts become `(2)` before generation; invalid
citations in the baseline dropped from 2.5% to 0%.

## 5. Hybrid search was *worse* than BM25 with a weak embedder
**What:** with the offline LSA embedder, hybrid recall@5 (0.72) was below
BM25 alone (0.79): equal-weight RRF lets a weak retriever push good BM25
hits down.
**Impact:** the ablation would have "proved" hybrid search hurts.
**Lesson:** fusion only helps when both retrievers are good; measure each one
alone (that is what the ablation is for) before blaming the fusion.

## 6. Our eval set favours keyword search
We wrote most questions while reading the passages, so they share words with
them. Real users phrase things differently (and some ask in French). Recall
numbers are optimistic until real questions from `logs/requests.jsonl` are
labelled and added.

## 7. Build environment
The development sandbox could not reach PyPI, HuggingFace or the Gemini API,
so the default embedder, reranker and generator could not be run there. We
kept an offline path (LSA embeddings, extractive generator, a Gemini-shaped
stub server in `tests/stub_gemini.py`) so the pipeline, evals and service
were still tested end to end; real numbers come from a machine with network
access.
