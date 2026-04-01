#!/bin/bash
# =========================================================================
# Wave-2D MC Dropout — Seed 1
# =========================================================================
set -e
cd "$(dirname "$0")"
mkdir -p results

echo "╔════════════════════════════════════════════════════╗"
echo "║  Wave-2D MC Dropout — Seed 1                       ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

python3 baseline_wave2d.py --method mc_dropout --dropout 0.1 --seed 1

echo ""
echo "========================================================="
echo "Done! Kết quả tại: results/wave2d_results_mc_dropout_seed1.json"
echo "========================================================="
