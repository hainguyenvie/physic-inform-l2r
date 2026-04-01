#!/bin/bash
# =========================================================================
# Cài đặt môi trường cho PRRS Experiments
# =========================================================================
set -e

# Di chuyển về thư mục gốc của project
cd "$(dirname "$0")"

echo "Đang cập nhật mã nguồn mới nhất..."
git pull origin main || echo "Cảnh báo: Không thể pull từ Github (có thể bạn chưa setup SSH key trên server, bỏ qua và cài luôn lib...)"

echo "Đang cài đặt các thư viện Python: PyTorch, Numpy, SciPy, Matplotlib, tqdm..."
# Dùng pip3 nếu pip không trỏ đúng tới python3
if command -v pip3 &> /dev/null
then
    pip3 install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo "========================================================="
echo "✅ Cài đặt thành công! Bây giờ bạn có thể chạy kịch bản:"
echo "cd PRRS"
echo "./run_mc_dropout_only.sh"
echo "========================================================="
