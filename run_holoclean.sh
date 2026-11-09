#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-iris}"
MISSING_PERCENT="${MISSING_PERCENT:-0.2}"
MISSING_MECHANISM="${MISSING_MECHANISM:-MCAR}"
ITERATION="${ITERATION:-1}"
SEED="${SEED:-42}"
MAX_MISSING_LENGTH="${MAX_MISSING_LENGTH:-4}"
METHOD_TIMEOUT_SECONDS="${METHOD_TIMEOUT_SECONDS:-3600}"

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-holo}"
DB_USER="${DB_USER:-holocleanuser}"

DEFAULT_SOURCE_DIR="$ROOT_DIR/imputation_algorithms/Holoclean/Holoclean"
HOLOCLEAN_SOURCE_DIR="${HOLOCLEAN_SOURCE_DIR:-$DEFAULT_SOURCE_DIR}"

if [[ ! -f "$HOLOCLEAN_SOURCE_DIR/holoclean.py" ]]; then
  echo "HoloClean source not found: $HOLOCLEAN_SOURCE_DIR" >&2
  echo "Set HOLOCLEAN_SOURCE_DIR to the directory containing holoclean.py, detect/, and repair/." >&2
  exit 2
fi

if [[ -z "${DB_PASSWORD:-}" ]]; then
  echo "DB_PASSWORD must be set for the HoloClean PostgreSQL connection." >&2
  exit 2
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "psql was not found on PATH. Install PostgreSQL client tools first." >&2
  exit 2
fi

if ! PGPASSWORD="$DB_PASSWORD" psql \
  --host "$DB_HOST" \
  --port "$DB_PORT" \
  --username "$DB_USER" \
  --dbname "$DB_NAME" \
  --command "SELECT 1;" >/dev/null 2>&1; then
  echo "Cannot connect to PostgreSQL at $DB_HOST:$DB_PORT/$DB_NAME as $DB_USER." >&2
  exit 2
fi

export DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD HOLOCLEAN_SOURCE_DIR

if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m code.workflow generate \
    --dataset "$DATASET" \
    --dataset_category no_real_missing \
    --iter "$ITERATION" \
    --seed "$SEED" \
    --missing_percent "$MISSING_PERCENT" \
    --max_missing_lenth "$MAX_MISSING_LENGTH" \
    --missing_mechanism "$MISSING_MECHANISM"
fi

"$PYTHON_BIN" -m code.workflow impute \
  --dataset "$DATASET" \
  --dataset_category no_real_missing \
  --iter "$ITERATION" \
  --seed "$SEED" \
  --missing_percent "$MISSING_PERCENT" \
  --max_missing_lenth "$MAX_MISSING_LENGTH" \
  --missing_mechanism "$MISSING_MECHANISM" \
  --imputation_method Holoclean \
  --method_timeout_seconds "$METHOD_TIMEOUT_SECONDS"
