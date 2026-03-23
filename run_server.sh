#!/bin/bash
# ============================================================
# PRRS — Server Run Script (GitHub clone workflow)
# Usage: bash run_server.sh [--mhd-only] [--ns-only]
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$REPO_ROOT/PRRS/results_all"
LOGS="$REPO_ROOT/logs"

mkdir -p "$RESULTS" "$LOGS"

echo "========================================================"
echo " PRRS Experiments — $(date)"
echo " REPO: $REPO_ROOT"
echo " RESULTS: $RESULTS"
echo "========================================================"

# ── 1. Install deps ───────────────────────────────────────────────────────────
echo ""
echo "==> Installing Python packages..."
pip install --quiet --break-system-packages torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet --break-system-packages numpy scipy matplotlib tqdm

# ── 2. Run experiments ────────────────────────────────────────────────────────

if [[ "$1" != "--mhd-only" ]]; then
    echo ""
    echo "==> [1/3] NS-2D — 5 seeds, n_train=500, normalized PRE"
    python3 "$REPO_ROOT/PRRS/prrs_ns2d.py" \
      --seeds 0 1 2 3 4 \
      --n-train 500 --n-cal 200 \
      --normalize-pre \
      --output-dir "$RESULTS" \
      2>&1 | tee "$LOGS/ns2d_5seeds.log"
fi

if [[ "$1" != "--ns-only" ]]; then
    echo ""
    echo "==> [2/3] MHD-2D — 3 seeds, 5 equations (fixed: per-var norm + grad clip)"
    python3 "$REPO_ROOT/PRRS/prrs_mhd2d.py" \
      --seeds 0 1 2 \
      --n-equations 5 \
      --output-dir "$RESULTS" \
      2>&1 | tee "$LOGS/mhd2d_3seeds.log"
fi

if [[ "$1" != "--mhd-only" && "$1" != "--ns-only" ]]; then
    echo ""
    echo "==> [3/3] Advection 1D — seed 42"
    python3 "$REPO_ROOT/PRRS/prrs_advection.py" \
      --seed 42 \
      2>&1 | tee "$LOGS/advection.log"
fi

echo ""
echo "================================================================"
echo " ALL DONE — results in $RESULTS"
echo "================================================================"
ls "$RESULTS"/*.json 2>/dev/null || echo "(no json yet)"
