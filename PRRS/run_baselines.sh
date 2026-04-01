#!/bin/bash
# =========================================================================
# RUN EXPERIMENTS: MC Dropout & Deep Ensemble baselines
# (For Wave-2D and NS-2D)
# =========================================================================
set -e

# Chạy từ thư mục PRRS
cd /home/h2n/l2r/CP-PRE/PRRS
mkdir -p results

echo "╔════════════════════════════════════════════════════╗"
echo "║ 1. WAVE-2D EXPERIMENTS                             ║"
echo "╚════════════════════════════════════════════════════╝"

# =============================================
# MC Dropout trên Wave-2D (Chạy 5 seeds)
# Setup: FNO Layers sẽ bật Dropout (p=0.1) sau GeLU, 
# Inference bằng cách chạy 20 Forward Passes ở Train Mode.
# =============================================
echo "----------------------------------------"
echo ">>> WAVE-2D: MC Dropout (p=0.1)"
echo "----------------------------------------"
for SEED in 0 42 1 2 3
do
    echo "[!] Training & Evaluating MC Dropout Wave-2D - Seed $SEED"
    python3 baseline_wave2d.py --method mc_dropout --dropout 0.1 --seed $SEED
done

# =============================================
# Deep Ensemble trên Wave-2D 
# Setup: Dùng 5 pretrained FNOs. Vì vậy hãy chắc chắn 
# 5 files prrs_wave2d_fno_seed*.pt (hoặc wave2d_fno_seed*.pt) đã được sinh ra.
# =============================================
echo "----------------------------------------"
echo ">>> WAVE-2D: Deep Ensemble Inference"
echo "----------------------------------------"
# Nếu chưa có 5 models độc lập (vì lý do nào đó), bạn phải chạy prrs_wave2d.py 
# cho từng seed: 0, 42, 1, 2, 3 để nó sinh ra file checkpoint. 
# Ensemble inference không cần Loop qua seed mà nó sẽ tự load đủ list seed.
python3 baseline_wave2d.py --method ensemble --seed 0


echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║ 2. NS-2D EXPERIMENTS                               ║"
echo "╚════════════════════════════════════════════════════╝"

# =============================================
# MC Dropout trên NS-2D (Chạy 5 seeds)
# =============================================
echo "----------------------------------------"
echo ">>> NS-2D: MC Dropout (p=0.1)"
echo "----------------------------------------"
for SEED in 0 1 2 3 4
do
    echo "[!] Training & Evaluating MC Dropout NS-2D - Seed $SEED"
    python3 baseline_ns2d.py --method mc_dropout --dropout 0.1 --seed $SEED
done

# =============================================
# Deep Ensemble trên NS-2D
# =============================================
echo "----------------------------------------"
echo ">>> NS-2D: Deep Ensemble Inference"
echo "----------------------------------------"
python3 baseline_ns2d.py --method ensemble --seed 0

echo "========================================================="
echo "All baseline experiments completed!"
echo "Kết quả, logs và hình ảnh Risk-Coverage đã được lưu tại thư mục PRRS/results"
echo "========================================================="
