# Báo Cáo Cấu Hình Kỹ Thuật: MC Dropout & Deep Ensemble Baselines cho mô hình Neural PDEs

_Tài liệu này ghi chú chi tiết phương pháp thiết lập hai baselines Bayesian Uncertainty (MC Dropout và Deep Ensemble) để so sánh trực tiếp với **Physics-Residual Rejection Score (PRRS)**. Được chuẩn bị nhằm giải quyết phản hồi từ Reviewers (NeurIPS) khi so sánh phương pháp giám sát bằng kiến thức vật lý (Physics-Informed) với các phương pháp Data-Driven UQ truyền thống._

---

## 1. Động Lực (Motivation)
Một câu hỏi gần như chắc chắn từ hội đồng phản biện (Reviewers) cho các bài báo về Learning to Reject (L2R) hoặc UQ trên Neural PDEs sẽ là: *"Tại sao không dùng các phương pháp Bayesian UQ cơ bản như MC Dropout hay Deep Ensemble làm chuẩn từ chối (Rejection Score) thay vì dùng dư lượng vật lý (PRE)?"*

Thực nghiệm này được sinh ra để cung cấp dữ liệu thực tế cứng rắn:
1. So sánh trên dữ liệu **In-Distribution (In-Dist)**: Phân tích xem tín hiệu bất định của mô hình (Model Uncertainty) có tốt hơn sai số vật lý (Physics Residual) hay không.
2. So sánh trên dữ liệu **Out-of-Distribution (OOD)**: Điểm mấu chốt để chứng minh rằng Bayesian UQ sẽ bị "mù" khi gặp OOD do phụ thuộc dữ liệu hiệu chuẩn (Data-dependent), trong khi PRRS mạnh mẽ vượt trội do không cần nhãn (Data-free Physics Firewall).

---

## 2. Thiết Lập Hệ Thống: MC Dropout

Để đảm bảo việc so sánh (fair comparison) không làm vỡ kiến trúc học chuỗi Fourier phức tạp của FNO gốc, MC Dropout được cấu hình một cách tinh vi như sau:

### 2.1. Vị Trí Cấy Gọi Dropout
Trong mô hình **Fourier Neural Operator (FNO2D)**, tín hiệu đi qua các khối xử lý: `Lift` $\rightarrow$ `6 x FNO Blocks` $\rightarrow$ `Projection`.
- **Không đặt Dropout ở**: `Lift`, `Projection` (các lớp tuyến tính đơn giản dễ mất mát thông tin), và `SpectralConv2d` (can thiệp làm nhiễu đổi không gian tần số Fourier phá vỡ ý nghĩa vật lý).
- **Vị trí tích hợp**: `nn.Dropout3d(p=dropout)` được đặt ở đầu ra cuối của mỗi **FNOBlock2d**, chính xác là **sau hàm kích hoạt GeLU và phép cộng skip-connection**. Điều này giúp tạo ra độ bất định (uncertainty) có ý nghĩa trên không gian tính năng phi tuyến mà mạng trích xuất.

### 2.2. Tham Số Training
- **Dropout Rate**: $p = 0.1$. Do FNO đã được tinh chỉnh bằng phổ cắt cụt (`modes=16` hoặc `modes=8`), một tỷ lệ rớt (dropout rate) cao hơn (như $0.2$ hoặc $0.5$) sẽ làm sụp đổ hoàn toàn hiệu suất của FNO, phá hỏng khả năng dự báo $L2$ baseline để có thể đem ra so sánh công bằng. 
- Mô hình **sẽ được Train lại** (Retrained) với cờ `dropout=0.1` được bật dựa trên cùng Hypereparameters của PRRS. Điều này giúp model học cách kháng tự nhiên với nhiễu bất định và Calibration sẽ mượt mà hơn. 

### 2.3. Quá Trình Inference (Sinh Điểm Bất Định)
Ở bước dự đoán (Inference / Calibration / Evaluation):
- Ép mô hình giữ chế độ huấn luyện: **`model.train()`** (không dùng `model.eval()`).
- Chạy K-vòng lặp tiến (Forward Passes): $K = 20$.
- **Lấy Điểm Sáng Rejection Score**: Với $K$ dự đoán cho ra shape `(K, batch, vars, Nx, Ny, T_out)`, hệ thống gom kỳ vọng và trích xuất độ lệch chuẩn (Standard Deviation) xuyên suốt trục $K$. Sau đó tính giá trị trung bình qua tất cả các trục vật lý (Tọa độ không gian và Thời gian) để thu gọn lại thành **1 Scalar Uncertainty Score** duy nhất trên mỗi mẫu (Sample).

---

## 3. Thiết Lập Hệ Thống: Deep Ensemble

Phương pháp Deep Ensemble tận dụng nguyên tắc "sự bất đồng giữa các mô hình độc lập" để đo lường độ bất định.

### 3.1. Training (Xây Dựng Tập Hợp)
- Hệ thống không cần train lại một kiến trúc mới mà sử dụng thẳng **5 FNO Models** được huấn luyện hoàn toàn độc lập với **5 Random Seeds** khác nhau (Ví dụ: Wave-2D sử dụng các seeds `0, 42, 1, 2, 3` | NS-2D sử dụng `0, 1, 2, 3, 4`).
- Không sử dụng Dropout ($p=0.0$). 

### 3.2. Inference
- Đọc 5 checkpoint `*fno_seed{X}.pt` vào memory trong chế độ `model.eval()`.
- Chạy chung một Batch dữ liệu qua cả 5 models, thu được một tensor gồm 5 dự đoán độc lập `(5, batch, vars, Nx, Ny, T_out)`.
- Giống MC Dropout, **Uncertainty Score (Rejection Score)** chính là độ lệch chuẩn (Standard Deviation) đếm trên 5 luồng dự đoán này. 

---

## 4. Tích Hợp Đánh Giá Ngưỡng Từ Chối (Threshold Framework)

Một khi có được `Uncertainty Score` từ MC Dropout hoặc Deep Ensemble (thay vì Physics PRE score), score này được cung cấp song song qua 2 thuật toán từ chối (Rejection methods) đang có trong PRRS Pipeline để đánh giá AUC-RC (Area Under Risk-Coverage Curve):

1. **MC-CP / Ensemble-CP (Dựa trên PRE-CP gốc)**: Tính toán phân vị $90\%$ của các `Uncertainty Scores` trên tập Calibration. Fixed constant quantile làm ngưỡng.
2. **MC-PRRS / Ensemble-PRRS (Dựa trên hệ Lagrangian Learning)**: Tối ưu hóa điểm `Uncertainty Scores` bằng Soft Sigmoid Surrogate, ép Coverage $\ge 90\%$ để tìm ra ngưỡng linh hoạt tối ưu $\tau^*$. 

> **Critical Note on OOD Threshold**: Ngưỡng $\tau^*$ hoặc $\hat{q}_\alpha$ đều được hiệu chuẩn (calibrated) hoàn toàn trên tập **In-distribution Calibration Set** và áp dụng nguyên trạng (as-is) lên dữ liệu OOD. Không có hành động re-calibration nào trên tập OOD. Thiết lập này tạo ra một sự so sánh công bằng (fair comparison) trực tiếp. Khả năng tự mở rộng của điểm PRRS (physics-based score generalizes to OOD without re-calibration) chính là lợi thế cốt lõi so với data-driven uncertainty của MC Dropout/Ensemble vốn không chắc chắn khi dùng threshold từ tập in-dist.

> Sự tích hợp ở cả Lựa Chọn 1 và 2 mang ý nghĩa luận điểm sắt đá: So sánh được "Score Vật Lý (PRE) so với Score Bất Định (Std)" trên cùng một sân chơi của PRE-CP và trên cùng sân chơi tối ưu Lagrangian của PRRS.

---

## 5. Cấu Trúc File & Cách Chạy (Execution) 

Tất cả đã cấu hình trong một Scripts Bash tự động hóa hoàn toàn luồng Evaluation này.

- **Các Tệp mã Nguồn Tách Biệt**:
  - `baseline_wave2d.py`: Quản lý logic Wave-2D (MC Dropout / Ensemble).
  - `baseline_ns2d.py`: Quản lý logic NS-2D (MC Dropout / Ensemble).
- **Tiện ích Command-Line**: Sử dụng các Argument Parses linh hoạt: `--method [mc_dropout, ensemble, prrs]` và `--dropout [float]`.

**Triển khai Thí Nghiệm (Run bash script)**: 
```bash
./run_baselines.sh
```

**Bảng Tiêu Chí Kỳ Vọng Sẽ Đạt Được Sau Thực Nghiệm**:
| Hạng mục Đo | Kỳ vọng từ MC Dropout/Ensemble | Kỳ vọng PRRS so sánh |
| :--- | :--- | :--- |
| **In-Distribution Baseline** | Rủi ro giảm vừa phải, có thể nhỉnh hơn chút xíu so với PRE nếu L2 FNO cực kỳ thấp. | Rủi ro ngang bằng nhau do cùng hội tụ trên Calibration Set. |
| **Out-of-Distribution (OOD)** | **SUY GIẢM HIỆU SUẤT ĐÁNG KỂ**: Không thể phân biệt OOD rõ ràng vì MC/Ensemble chỉ tin vào dữ liệu Training gốc. Cụ thể, MC Dropout và Deep Ensemble có OOD rejection X% và Y% respectively, significantly lower than PRRS's 100%. | **VƯỢT TRỘI / KHÔNG THỂ BÀN CÃI**: Trực tiếp đo dư lượng phương trình vật lý, 100% Guardrail chặn đứng các sai trái vật lý, Coverage OOD về $0\%$. |

_Báo cáo này có thể được Copy/Paste trực tiếp làm sườn (Outline) hoặc Appendix cho mô tả các phương pháp Baseline UQ trong bản thảo bài báo._
