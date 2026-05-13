#!/usr/bin/env bash
# Minimal end-to-end smoke test for the retriever CLI:
#
#   1. ingest one (or more) PDF(s) into a temp LanceDB table,
#   2. inspect the table to confirm rows were written,
#   3. (optionally) compute recall@1/5/10 against a ground-truth CSV,
#   4. clean up.
#
# Usage:
#   recall_smoke_test.sh <pdf_or_dir> [query_csv]
#
# Examples:
#   ./recall_smoke_test.sh data/multimodal_test.pdf
#   ./recall_smoke_test.sh data/pdfs/ data/bo767_query_gt.csv
#
# Exits 0 only if every step succeeds. Cleans up the temp LanceDB on exit.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pdf_or_dir> [query_csv]" >&2
  exit 64
fi

PDF_INPUT="$1"
QUERY_CSV="${2:-}"

if [[ ! -e "$PDF_INPUT" ]]; then
  echo "ERROR: input path does not exist: $PDF_INPUT" >&2
  exit 1
fi

if ! command -v retriever >/dev/null 2>&1; then
  echo "ERROR: 'retriever' CLI not in PATH. Activate the nemo-retriever venv." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSPECT_PY="$SCRIPT_DIR/inspect_lancedb.py"

WORKDIR="$(mktemp -d -t retriever-smoke.XXXXXX)"
LANCEDB_URI="$WORKDIR/lancedb"
TABLE_NAME="smoke-test"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "==> Smoke test workdir: $WORKDIR"
echo "==> Step 1/3: ingesting $PDF_INPUT into $LANCEDB_URI/$TABLE_NAME"
retriever ingest "$PDF_INPUT" \
  --lancedb-uri "$LANCEDB_URI" \
  --table-name "$TABLE_NAME"

echo
echo "==> Step 2/3: inspecting the table"
if [[ -f "$INSPECT_PY" ]]; then
  python "$INSPECT_PY" "$LANCEDB_URI" "$TABLE_NAME" --samples 2
else
  echo "WARN: inspect_lancedb.py not found at $INSPECT_PY; skipping inspection."
fi

if [[ -z "$QUERY_CSV" ]]; then
  echo
  echo "==> Step 3/3: no query CSV provided; running a smoke query instead"
  retriever query "what is on page 1?" \
    --lancedb-uri "$LANCEDB_URI" \
    --table-name "$TABLE_NAME" \
    --top-k 3
  echo
  echo "Smoke test passed (ingest + inspect + 1 query). No recall metrics."
  exit 0
fi

if [[ ! -f "$QUERY_CSV" ]]; then
  echo "ERROR: query CSV does not exist: $QUERY_CSV" >&2
  exit 1
fi

echo
echo "==> Step 3/3: computing recall against $QUERY_CSV"
retriever recall vdb-recall run \
  --query-csv "$QUERY_CSV" \
  --lancedb-uri "$LANCEDB_URI" \
  --table-name "$TABLE_NAME" \
  --top-k 10 \
  --no-print-hits

echo
echo "Smoke test passed (ingest + inspect + recall)."
