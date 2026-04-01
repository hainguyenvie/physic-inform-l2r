#!/bin/bash
# =========================================================================
# Cài đặt môi trường cho PRRS Experiments
# =========================================================================
set -e

# Di chuyển về thư mục gốc của project
cd "$(dirname "$0")"

echo "Đang cập nhật mã nguồn mới nhất..."
git pull origin main || echo "Cảnh báo: Không thể pull từ Github (có thể bạn chưa setup SSH key trên server, bỏ qua và cài luôn lib...)"

echo "Đang kiểm tra và cài pip..."
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null
then
    echo "pip chưa được cài đặt, tiến hành tự động cài đặt qua apt/yum..."
    if command -v apt-get &> /dev/null
    then
        apt-get update && apt-get install -y python3-pip
    elif command -v yum &> /dev/null
    then
        yum install -y python3-pip
    else
        echo "Lỗi: Không tìm thấy trình quản lý gói để tự cài pip. Vui lòng cài đặt python3-pip thủ công!"
        exit 1
    fi
fi

echo "Đang cài đặt các thư viện Python: PyTorch, Numpy, SciPy, Matplotlib, tqdm..."
# Dùng pip3 nếu pip không trỏ đúng tới python3
if command -v pip3 &> /dev/null
then
    pip3 install --break-system-packages -r requirements.txt || pip3 install -r requirements.txt
else
    pip install --break-system-packages -r requirements.txt || pip install -r requirements.txt
fi

echo "========================================================="
echo "✅ Cài đặt thành công! Bây giờ bạn có thể chạy kịch bản:"
echo "cd PRRS"
echo "./run_mc_dropout_only.sh"
echo "========================================================="
