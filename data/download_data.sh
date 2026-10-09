#!/usr/bin/env bash
# Downloads NSL-KDD (KDDTrain+.txt, KDDTest+.txt) into this folder. Data is not committed to the repo.
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
for f in KDDTrain+.txt KDDTest+.txt; do
  [ -f "$f" ] || curl -fsSL "$BASE/$f" -o "$f"
done
echo "NSL-KDD ready in $(pwd)"
