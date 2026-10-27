"""Check every gold label before trusting any metric built on it.

For each answerable question and each source:
  1. the file exists in the corpus,
  2. the evidence span really occurs in that file,
  3. the evidence survives cleaning + chunking, i.e. at least one indexed
     chunk of that file contains it (otherwise recall can never reach 1).

    python eval/validate_qa.py [--index index]
"""
import argparse
from pathlib import Path

from common import chunk_matches, load_qa, norm

from librarian.config import settings
from librarian.index import Index

ap = argparse.ArgumentParser()
ap.add_argument("--index", default=settings.index_dir)
args = ap.parse_args()

qa = load_qa()
ix = Index(args.index)
by_path: dict[str, list] = {}
for c in ix.chunks:
    by_path.setdefault(c.path, []).append(c)

root = Path(settings.corpus_dir)
problems = 0
for item in qa:
    for s in item["sources"]:
        f = root / s["path"]
        if not f.exists():
            print(f"{item['id']}: missing file {s['path']}"); problems += 1; continue
        if norm(s["evidence"]) not in norm(f.read_text(errors="replace")):
            print(f"{item['id']}: evidence not in file {s['path']}: {s['evidence'][:60]!r}"); problems += 1; continue
        if not any(chunk_matches(c, s) for c in by_path.get(s["path"], [])):
            print(f"{item['id']}: evidence split or dropped by chunker in {s['path']}: {s['evidence'][:60]!r}")
            problems += 1

n_ans = sum(q["answerable"] for q in qa)
print(f"{len(qa)} questions ({n_ans} answerable, {len(qa) - n_ans} unanswerable); {problems} problems")
raise SystemExit(1 if problems else 0)
