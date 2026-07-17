#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${DEVELOPMENT_STATS:?Set DEVELOPMENT_STATS to the development stats directory}"
: "${EXTERNAL_STATS:?Set EXTERNAL_STATS to the external ablation stats directory}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" scripts/htsr_combined_study.py \
  --metadata data/sampled_metadata.csv \
  --labels data/manual_audits/spectral_family_retrieval_label_freeze.csv \
  --graph fetch_data/large_nlp_data.json \
  --development-stats "$DEVELOPMENT_STATS" \
  --development-targets Model-Zoo/data/curated/model_zoo_phase2.csv \
  --external-stats "$EXTERNAL_STATS" \
  --external-targets Model-Zoo/data/ablation/ablation_selected.csv \
  --cache-root research_runs/htsr_study/cache/study_features \
  --output-root research_runs/htsr_study/runs \
  --protocol configs/htsr_combined_study_v3.json \
  "$@"
