#!/usr/bin/env bash
# Bootstrap the TimescaleDB schema manually (if not using docker-compose initdb.d)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-stocks}"
DB_USER="${DB_USER:-postgres}"

echo "Bootstrapping database $DB_NAME on $DB_HOST:$DB_PORT..."

PGPASSWORD="${DB_PASSWORD:-stockpass}" psql \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -f "$SCRIPT_DIR/bootstrap_db.sql"

echo "Database bootstrap complete."
