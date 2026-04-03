<!--
  Sync Impact Report
  ==================
  Version change: 0.0.0 (template) → 1.0.0
  Modified principles: N/A (initial population from template)
  Added sections:
    - Core Principles (7 principles)
    - Data Integrity & Leakage Prevention
    - Development Workflow
    - Governance
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md — ✅ reviewed, no changes needed
    - .specify/templates/spec-template.md — ✅ reviewed, no changes needed
    - .specify/templates/tasks-template.md — ✅ reviewed, no changes needed
  Follow-up TODOs: None
-->

# Stock Forecast ML Pipeline Constitution

## Core Principles

### I. No Data Leakage (NON-NEGOTIABLE)
Data leakage is the single biggest failure mode in financial ML.
Every pipeline step MUST enforce temporal causality:
- Feature normalization (scalers) MUST be fitted INSIDE each
  walk-forward fold, never on the full dataset.
- Macro data MUST use FRED ALFRED vintage (as-reported) values,
  not retroactively revised data.
- A purged embargo gap of at least 21 calendar days MUST separate
  the training window end and the test window start (matching the
  maximum forecast horizon of 20 trading days).
- Forward-return targets MUST be computed from future prices; no
  feature column may reference data after the observation timestamp.

### II. Walk-Forward Validation Only
Random train/test splits are prohibited for any time-series model
evaluation. All model training and evaluation MUST use walk-forward
(expanding or sliding window) validation as implemented in
`src/training/walk_forward.py`. The validation configuration
(initial train years, test months, step months, gap days) is
defined in `configs/data_config.yaml` and MUST NOT be overridden
ad-hoc in scripts or notebooks.

### III. Config-Driven, Single Source of Truth
All hyperparameters, ticker universes, feature settings, data
sources, and validation parameters MUST live in the three YAML
config files (`configs/tickers.yaml`, `configs/data_config.yaml`,
`configs/model_config.yaml`). Hard-coded magic numbers in source
modules are prohibited. Environment-specific values (credentials,
host addresses) MUST be resolved via `.env` and the config loader
at `src/utils/config.py`.

### IV. Idempotent and Reproducible Pipelines
Every collector, feature builder, and training run MUST be
idempotent — safe to re-run without duplicating data or corrupting
state. Database writes MUST use upsert (ON CONFLICT) semantics.
All random processes MUST use the seed defined in
`configs/model_config.yaml` (currently `42`). The combination of
config files + git commit MUST be sufficient to reproduce any
historical result.

### V. Graceful Degradation
The pipeline MUST operate in reduced-capability mode when optional
components are unavailable:
- No FRED API key → prices-only mode (skip macro features).
- No vLLM endpoint → FinBERT-only sentiment (skip LLM scorer).
- No GPU → CPU-only PyTorch (slower but functional).
- Database down → parquet-file fallback for reads.
Each collector and feature module MUST log a warning and continue,
never crash the full pipeline for a single missing source.

### VI. Adaptive Targets Over Fixed Thresholds
Classification targets (up/flat/down) MUST use volatility-adaptive
thresholds (`k * sqrt(horizon) * rolling_std`), not fixed
percentage cutoffs. This ensures stable class distributions across
market regimes (bull, bear, crisis). The threshold parameter `k`
and lookback window are defined in `configs/data_config.yaml`.

### VII. Model Comparison Before Promotion
No model may be promoted to production use or serve live
predictions without walk-forward evaluation results saved as JSON
in the `models/` directory. The `compare-models` Makefile target
MUST show the new model alongside all existing baselines.
Directional accuracy (up vs down, ignoring flat) is the primary
ranking metric; Sharpe ratio is the secondary trading metric.

## Data Integrity & Leakage Prevention

The following constraints are specific to financial time-series
data and MUST be enforced in every new module:

- **No future data in features.** Any rolling calculation on a
  feature column MUST use only past values (`rolling`, `expanding`,
  `shift`). Using `.mean()` or `.std()` without a window implies
  full-dataset statistics and is a leakage vector.
- **Per-ticker isolation.** Cross-sectional features are computed
  across the universe at each timestamp, but never mix data from
  different dates within a single row.
- **Parquet + DB dual storage.** Raw data is saved to parquet
  (fast local reads, no DB dependency) AND upserted to TimescaleDB
  (SQL queries, continuous aggregates). Both MUST stay in sync.
- **Vintage macro data.** The `macro_collector` uses FRED ALFRED
  API to fetch the value that was publicly known at each date, not
  the latest revised value. The USD index is spliced from DTWEXB
  (pre-2006) and DTWEXBGS (2006+).

## Development Workflow

1. **New features start with a config entry.** If a new indicator,
   data source, or model needs parameters, add them to the
   appropriate YAML config file first.
2. **Makefile is the interface.** Every repeatable action MUST have
   a Makefile target. Users should never need to remember raw
   `python -m` commands.
3. **Backward-compatible collectors.** Adding a new ticker or
   indicator MUST NOT require re-running the full backfill. The
   daily update script handles incremental additions.
4. **Dashboard reflects API.** Every metric visible in the Next.js
   dashboard MUST come from a documented FastAPI endpoint. No
   direct database queries from the frontend.
5. **SPEC.md stays current.** When a phase is completed or a
   design decision changes, `SPEC.md` MUST be updated in the same
   commit. It is the authoritative record of what exists and why.

## Governance

This constitution supersedes ad-hoc practices. Amendments require:
1. A clear description of the change and its rationale.
2. Verification that the change does not introduce data leakage
   (Principle I).
3. Update to this file, SPEC.md, and any affected templates.
4. Version bump following semantic versioning:
   - MAJOR: Principle removed or redefined in a breaking way.
   - MINOR: New principle added or section materially expanded.
   - PATCH: Wording clarifications or typo fixes.

All code changes MUST comply with these principles. Complexity
beyond what the task requires MUST be justified in a PR
description. Use `SPEC.md` as the reference specification and
`GETTING_STARTED.md` as the runtime development guide.

**Version**: 1.0.0 | **Ratified**: 2026-03-31 | **Last Amended**: 2026-03-31
