#!/usr/bin/env bash
# Fetch the corpus: the Documentation/ tree of one pinned kernel release.
#
# * --sparse + sparse-checkout: only Documentation/ is checked out (~85 MB),
#   not the 1.5 GB source tree.
# * --filter=blob:none --shallow-since: we keep commit history back to
#   $SINCE (but no old file contents) so every document gets a real
#   "last changed" date for its metadata. Files untouched since $SINCE are
#   marked as such rather than given a fake date.
set -euo pipefail
TAG="${KERNEL_TAG:-v7.3}"
SINCE="${SINCE:-2021-01-01}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/corpus"
mkdir -p "$DIR"

if [ ! -d "$DIR/linux/.git" ]; then
  echo "[corpus] cloning linux $TAG (Documentation/ only, history since $SINCE)…"
  git clone -q --branch "$TAG" --filter=blob:none --shallow-since="$SINCE" \
      --sparse https://github.com/torvalds/linux.git "$DIR/linux"
  git -C "$DIR/linux" sparse-checkout set Documentation
else
  echo "[corpus] $DIR/linux already present"
fi

echo "[corpus] computing last-changed dates per file…"
git -C "$DIR/linux" log --format='>%H %cs' --name-only --no-renames HEAD -- Documentation \
  | python3 "$(dirname "$0")/dates_from_log.py" "$SINCE" "$DIR/linux/.git/shallow" > "$DIR/dates.json"
echo "[corpus] done: $(find "$DIR/linux/Documentation" -type f | wc -l) files"
