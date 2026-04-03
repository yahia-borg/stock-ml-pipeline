#!/usr/bin/env bash
# First-time project setup — run once after cloning
# Usage: bash scripts/setup.sh [--gpu]
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

GPU_MODE=false
if [[ "${1:-}" == "--gpu" ]]; then
    GPU_MODE=true
fi

info()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo "========================================="
echo "  Stock Forecast Pipeline — Setup"
echo "========================================="
echo ""

# ── 1. Check prerequisites ──
echo "Checking prerequisites..."

# Python 3.11+
if command -v python3.11 &>/dev/null; then
    PY=python3.11
elif command -v python3 &>/dev/null; then
    PY=python3
    PY_VER=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$(echo "$PY_VER < 3.11" | bc -l 2>/dev/null || echo 1)" == "1" ]]; then
        PY_MAJOR=$($PY -c 'import sys; print(sys.version_info.major)')
        PY_MINOR=$($PY -c 'import sys; print(sys.version_info.minor)')
        if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ]]; then
            fail "Python 3.11+ required, found $PY_VER. Install it: https://www.python.org/downloads/"
        fi
    fi
else
    fail "Python 3 not found. Install Python 3.11+: https://www.python.org/downloads/"
fi
info "Python: $($PY --version)"

# Docker
if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install it: https://docs.docker.com/get-docker/"
fi
if ! docker info &>/dev/null 2>&1; then
    fail "Docker daemon not running. Start Docker Desktop or run: sudo systemctl start docker"
fi
info "Docker: $(docker --version | head -1)"

# Docker Compose
if docker compose version &>/dev/null 2>&1; then
    info "Docker Compose: $(docker compose version --short)"
elif command -v docker-compose &>/dev/null; then
    warn "Found docker-compose (v1). Consider upgrading to Docker Compose v2."
else
    fail "Docker Compose not found."
fi

# Git
if command -v git &>/dev/null; then
    info "Git: $(git --version)"
else
    warn "Git not found — you won't be able to push/pull changes."
fi

echo ""

# ── 2. Create virtual environment ──
if [[ ! -d .venv ]]; then
    echo "Creating virtual environment..."
    $PY -m venv .venv
    info "Virtual environment created at .venv/"
else
    info "Virtual environment already exists."
fi

source .venv/bin/activate

# ── 3. Install dependencies ──
echo "Upgrading pip..."
pip install --upgrade pip -q

echo "Installing PyTorch..."
if $GPU_MODE; then
    echo "  (GPU mode — downloading CUDA-enabled PyTorch, this may be large)"
    pip install torch --index-url https://download.pytorch.org/whl/cu124 -q
else
    echo "  (CPU mode — lightweight install)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi

echo "Installing project dependencies..."
pip install -e ".[dl,dev]" -q
info "All dependencies installed."

echo ""

# ── 4. Environment file ──
if [[ ! -f .env ]]; then
    cp .env.example .env
    info "Created .env from .env.example"
    warn "Edit .env to add your API keys (FRED, NewsAPI, etc.)"
else
    info ".env already exists."
fi

echo ""

# ── 5. Start database ──
echo "Starting TimescaleDB..."
docker compose up -d timescaledb

echo "Waiting for database to be ready..."
RETRIES=30
until docker compose exec timescaledb pg_isready -U postgres -q 2>/dev/null; do
    RETRIES=$((RETRIES - 1))
    if [[ $RETRIES -le 0 ]]; then
        fail "Database did not start in time. Check: docker compose logs timescaledb"
    fi
    sleep 1
done
info "TimescaleDB is running on localhost:${DB_PORT:-5432}"

echo ""

# ── 6. Verify ──
echo "Running health checks..."
python -c "from src.utils.db import check_connection; assert check_connection(), 'DB connection failed'"
info "Database connection verified."

python -c "from src.utils.config import get_ticker_list; print(f'  Tickers loaded: {len(get_ticker_list())}')"
info "Config loading verified."

echo ""
echo "========================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. source .venv/bin/activate"
echo "    2. Edit .env with your API keys"
echo "    3. make check-health"
echo ""
echo "  Useful commands:"
echo "    make help        — list all commands"
echo "    make test        — run tests"
echo "    make db-stop     — stop database"
echo "    make db          — start database"
echo "========================================="
