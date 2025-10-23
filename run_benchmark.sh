#!/usr/bin/env bash
set -euo pipefail

# Defaults
MAX_N=${MAX_N:-25}
M=${M:-5}
INSERT_BATCH=${INSERT_BATCH:-1000}
REPEATS=${REPEATS:-3}
FEW_REPEATS=${FEW_REPEATS:-2}
OUT_DIR=${OUT_DIR:-"$(pwd)/results"}
BRING_DOWN=${BRING_DOWN:-0}
CAT_MODE=${CAT_MODE:-""}  # empty|short|full|both

usage() {
  cat <<EOF
Usage: $(basename "$0") [-n MAX_N] [-m M] [-b INSERT_BATCH] [-r REPEATS] [-f FEW_REPEATS] [-o OUT_DIR] [--down] [--cat[=MODE]]

Options:
  -n MAX_N         Highest N (rows = 2^N). Default: ${MAX_N}
  -m M             Number of indexed columns (incl. PK). Default: ${M}
  -b INSERT_BATCH  Rows per insert batch for timing. Default: ${INSERT_BATCH}
  -r REPEATS       Number of repeated measurements per cold/hot metric. Default: ${REPEATS}
  -f FEW_REPEATS   Reduced repeats when first timings are fast (< TIME_THRESHOLD_S). Default: ${FEW_REPEATS}
  -o OUT_DIR       Host directory to copy results into. Default: ${OUT_DIR}
  --cat[=MODE]     Print results to stdout via cat. MODE: short|full|both (default: short if no MODE specified)
  --down           Bring down docker compose stack after run.

Environment overrides: MAX_N, M, INSERT_BATCH, REPEATS, FEW_REPEATS, OUT_DIR, BRING_DOWN=1, CAT_MODE
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) MAX_N="$2"; shift 2;;
    -m) M="$2"; shift 2;;
    -b) INSERT_BATCH="$2"; shift 2;;
  -o) OUT_DIR="$2"; shift 2;;
  -r) REPEATS="$2"; shift 2;;
  -f) FEW_REPEATS="$2"; shift 2;;
    --cat) CAT_MODE="short"; shift;;
    --cat=*) CAT_MODE="${1#*=}"; shift;;
    --down) BRING_DOWN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

mkdir -p "$OUT_DIR"
# Normalize OUT_DIR to an absolute path for reliable Docker bind mounts
case "$OUT_DIR" in
  /*) ;; # already absolute
  *) OUT_DIR="$(cd "$OUT_DIR" && pwd -P)" ;;
esac

# Export for docker compose variable substitution (used in bench service volume)
export BENCH_OUT_DIR="$OUT_DIR"

echo "[1/4] Building images..."
docker compose build

echo "[2/4] Starting Postgres (db) ..."
docker compose up -d db

# Wait for health if available (optional best effort)
for i in $(seq 1 30); do
  if docker inspect -f '{{.State.Health.Status}}' pgbench-db 2>/dev/null | grep -q healthy; then
    break
  fi
  sleep 1
done

echo "[3/4] Running benchmark (MAX_N=${MAX_N}, M=${M}, INSERT_BATCH=${INSERT_BATCH}, REPEATS=${REPEATS}, FEW_REPEATS=${FEW_REPEATS}) ..."
docker compose run --rm \
  -e MAX_N="${MAX_N}" \
  -e M_MAX="${M}" \
  -e INSERT_BATCH="${INSERT_BATCH}" \
  -e REPEATS="${REPEATS}" \
  -e FEW_REPEATS="${FEW_REPEATS}" \
  bench

echo "[4/4] Results are written directly to ${OUT_DIR} after each (N,M) combination. Listing contents:"
ls -la "${OUT_DIR}" || true

echo "Done. Results (updated incrementally):\n  ${OUT_DIR}/results.json\n  ${OUT_DIR}/results_full.json"

# Optionally cat results per request
case "${CAT_MODE}" in
  short)
    echo "\n===== results.json =====" && cat "${OUT_DIR}/results.json" || true
    ;;
  full)
    echo "\n===== results_full.json =====" && cat "${OUT_DIR}/results_full.json" || true
    ;;
  both)
    echo "\n===== results.json =====" && cat "${OUT_DIR}/results.json" || true
    echo "\n===== results_full.json =====" && cat "${OUT_DIR}/results_full.json" || true
    ;;
  "")
    ;; # no-op
  *)
    echo "Unknown CAT_MODE='${CAT_MODE}'. Use short|full|both." >&2
    ;;
esac

if [[ "$BRING_DOWN" == "1" ]]; then
  echo "Bringing down docker compose stack..."
  docker compose down -v
fi
