#!/bin/bash
# =========================================================================
# RERUN NS-2D MC Dropout với n_train=500, n_cal=200 (match PRRS v6 config)
# Fix blocking issue: baseline trước dùng n_train=200 sai với PRRS config
# =========================================================================
set -e
cd "$(dirname "$0")"
mkdir -p results

echo "╔════════════════════════════════════════════════════╗"
echo "║  NS-2D MC Dropout — RERUN (n_train=500, n_cal=200) ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Config: n_train=500, n_cal=200 — match PRRS v6 (results_summary.md Section 2.3)"
echo ""

# Seed 0 — rerun để replace kết quả cũ
echo ">>> NS-2D MC Dropout Seed 0 (rerun, fixed config)"
python3 baseline_ns2d.py --method mc_dropout --dropout 0.1 --seed 0 \
    --n-train 500 --n-cal 200

echo ""
# Seed 1 — chạy thêm theo yêu cầu
echo ">>> NS-2D MC Dropout Seed 1 (new)"
python3 baseline_ns2d.py --method mc_dropout --dropout 0.1 --seed 1 \
    --n-train 500 --n-cal 200

echo ""
echo "========================================================="
echo "Done! Kết quả tại: results/"
echo "  ns2d_results_mc_dropout_seed0.json  (replaced)"
echo "  ns2d_results_mc_dropout_seed1.json  (new)"
echo "========================================================="
