# ============================================================
# Multi-stage build for minimal production image
# Final image: ~600MB (vs ~2.5GB single-stage)
# ============================================================

# ── Stage 1: Build dependencies ──
FROM python:3.12-slim AS builder

WORKDIR /build

# Build tools (only in builder, not in final image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install deps into a virtual env we can copy to final stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU PyTorch (smallest variant)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project deps (cached unless pyproject.toml changes)
COPY pyproject.toml .
RUN mkdir -p src/utils && touch src/__init__.py src/utils/__init__.py && \
    pip install --no-cache-dir .


# ── Stage 2: Minimal runtime image ──
FROM python:3.12-slim AS runtime

WORKDIR /app

# Only runtime C libs (no compiler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/* && \
    # Create non-root user
    useradd -r -s /bin/false appuser

# Copy venv from builder (all Python packages)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create data/model dirs
RUN mkdir -p data/raw data/processed models logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8099

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8099/health || exit 1

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "2"]
