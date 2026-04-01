#!/bin/bash
# =========================================================================
# RUN EXPERIMENTS: MC Dropout baseline ONLY
# (For Wave-2D and NS-2D)
# =========================================================================
set -e

# Di chuyển vào folder chứa script
cd "$(dirname "$0")"
mkdir -p results

echo "╔════════════════════════════════════════════════════╗"
echo "║ 1. WAVE-2D EXPERIMENTS                             ║"
echo "╚════════════════════════════════════════════════════╝"

echo "----------------------------------------"
echo ">>> WAVE-2D: MC Dropout (p=0.1)"
echo "----------------------------------------"
for SEED in 0 42 1 2 3
do
    echo "[!] Training & Evaluating MC Dropout Wave-2D - Seed $SEED"
    python3 baseline_wave2d.py --method mc_dropout --dropout 0.1 --seed $SEED
done


echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║ 2. NS-2D EXPERIMENTS                               ║"
echo "╚════════════════════════════════════════════════════╝"

echo "----------------------------------------"
echo ">>> NS-2D: MC Dropout (p=0.1)"
echo "----------------------------------------"
for SEED in 0 1 2 3 4
do
    echo "[!] Training & Evaluating MC Dropout NS-2D - Seed $SEED"
    python3 baseline_ns2d.py --method mc_dropout --dropout 0.1 --seed $SEED
done

echo "========================================================="
echo "MC Dropout experiments completed!"
echo "Kết quả, logs và hình ảnh Risk-Coverage đã được lưu tại thư mục PRRS/results"
echo "========================================================="
