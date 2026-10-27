"""Read `git log --format='>%H %cs' --name-only` on stdin, print {path: date}.

git log is newest-first, so the first date seen for a path is its last change.
In a shallow clone the boundary commits (listed in .git/shallow) have no
parents, so git shows them "adding" every file. A file first seen at a
boundary commit was not changed since the cut-off: we record "<SINCE"
instead of inventing a date.
"""
import json
import sys

since, shallow_file = sys.argv[1], sys.argv[2]
with open(shallow_file) as f:
    boundary = {line.strip() for line in f if line.strip()}
dates, cur = {}, None
for line in sys.stdin:
    line = line.strip()
    if line.startswith(">"):
        sha, day = line[1:].split()
        cur = f"<{since}" if sha in boundary else day
    elif line and line not in dates:
        dates[line] = cur
out = {}
for path, d in dates.items():
    if not path.startswith("Documentation/"):
        continue
    out[path[len("Documentation/"):]] = d
json.dump(out, sys.stdout, indent=0, sort_keys=True)
