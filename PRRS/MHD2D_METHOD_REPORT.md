# MHD-2D PRRS — Method Report & Issues

*Generated: 2026-03-23*

---

## 1. Problem Setup

### PDE: 2D Incompressible Resistive MHD

Hai phương trình chính trong không gian vật lý (incompressible, 2D):

```
∂ω/∂t = J(ψ, ω) − J(A, j) + ν∇²ω        [vorticity + Lorentz force]
∂A/∂t = J(ψ, A) + η∇²A                   [magnetic induction]
```

Với các quan hệ:
- `J(f,g) = f_x·g_y − f_y·g_x`  (Poisson bracket / Jacobian)
- `ω = −∇²ψ`  →  stream function `ψ̂ = ω̂/K²`
- `j = −∇²A`  →  current density `ĵ = K²·Â`
- `u = ∂ψ/∂y`, `v = −∂ψ/∂x`  (velocity)
- `Bx = ∂A/∂y`, `By = −∂A/∂x`  (B field from magnetic potential)

**Tham số vật lý:** ν = η = 0.01

**Domain:** `[0, L]² = [0, 1]²`, N = 64, dx = 1/64 ≈ 0.0156

---

## 2. Numerical Solver

**Phương pháp:** Pseudo-spectral, Crank-Nicolson cho diffusion, explicit Jacobian

**Time stepping:**
- dt = 0.005, tEnd = 0.5 → Nt = 100 bước nội tại
- Lưu mỗi 5 bước → dt_sub = 0.025, Nt_sub = 20 frames + 1 IC = 21 frames

**Dealiasing:** 2/3 rule (giữ mode `|k| ≤ N/3` trong mỗi chiều)

**Crank-Nicolson update:**
```
ω̂^{n+1} = (numer_ω · ω̂^n + dt · Jw) / denom_ω
Â^{n+1}  = (numer_A · Â^n  + dt · J_ind) / denom_A
```
với `numer/denom = 1 ∓ 0.5·ν·K²·dt`

**NaN clipping:** Mỗi bước đều clip `oh, Ah` về 0 nếu không finite (để tránh NaN lan truyền với OOD amplitudes cao).

---

## 3. Initial Conditions

**Parameterized Orszag-Tang vortex** (benchmark chuẩn MHD, Section 5.3 paper):

```
k1 = 2π/L,  k2 = 4π/L (= 2·k1)

ω₀ = av · k1 · (cos(k1·x) + cos(k1·y))
A₀ = aB · (cos(k1·y)/k1 + cos(k2·x)/k2)

→ Bx = ∂A/∂y = −aB·sin(k1·y)
→ By = −∂A/∂x = aB·sin(k2·x)·(k2/k1)/k2 = aB·sin(k2·x)/(2)
```

**Ý nghĩa:** Năng lượng tập trung ở mode k=1,2 (low-frequency), dễ học hơn cho FNO so với random Fourier IC.

**Phân phối tham số:**

| Tập | av (vorticity amp) | aB (magnetic amp) |
|-----|-------------------|-------------------|
| In-dist (train/cal/val) | U[0.5, 1.5] | U[0.2, 0.8] |
| OOD | U[1.5, 2.0] | U[0.8, 1.2] |

---

## 4. Dataset

| Tập | N mẫu | Mô tả |
|-----|--------|-------|
| Train | 1000 | In-dist, fit normalizer |
| Cal | 400 | In-dist, conformal calibration |
| Val | 100 | In-dist, evaluation |
| OOD | 100 | Higher amplitude, out-of-distribution |

**Tensor shape:** `(N, 4, 64, 64, 21)` — biến [ω, A, u, v], spatial 64×64, thời gian 21 frames

---

## 5. Normalisation

**PerVarMinMaxNorm:** Normalize từng biến [ω, A, u, v] độc lập:
```
x_norm = 2·(x − lo_i) / (hi_i − lo_i + ε) − 1  ∈ [−1, 1]
```
- Fit trên training data, apply cho cal/val/OOD
- Tách `in_norm` (input frame T_in=1) và `out_norm` (output T_out=20 frames)

---

## 6. FNO Architecture

**FNO2D với n_vars=4 (multi-variable):**

| Component | Chi tiết |
|-----------|---------|
| Input | `(BS, 4, 64, 64, T_in=1)` + positional grid (x,y) |
| Lift | `Linear(T_in+2, width=32)` |
| Backbone | 6× `FNOBlock2d` với skip connections (f0→f2: skip, f3→f5: skip) |
| SpectralConv2d | rfft2 → multiply Fourier weights (modes=16×16) → irfft2 |
| Proj | `Linear(32, 128)` → GELU → `Linear(128, step=4)` |
| Output | `(BS, 4, 64, 64, step=4)` |
| Params | **12,607,364** (~12.6M) |

**Skip connections:**
```
x0 = f0(x)
x  = f2(f1(x0)) + x0      ← skip 1
x1 = f3(x)
x  = f5(f4(x1)) + x1      ← skip 2
```

---

## 7. Training

**Auto-regressive (AR) loop:**
- step=4: mỗi forward pass predict 4 frames liên tiếp
- T_out=20: cần 5 forward passes per batch
- `inp` sau mỗi step: `cat([inp[..., step:], out.detach()], dim=-1)[..., -T_in:]`

**Loss:** Relative L2 loss tổng qua 5 AR steps:
```
L = Σ_{t=0,4,8,12,16} ‖pred_t − true_t‖₂ / ‖true_t‖₂
```

**Optimizer:** Adam, lr=1e-3, weight_decay=1e-4

**LR schedule:** StepLR, giảm ×0.5 mỗi 100 epochs

**Gradient clipping:** `clip_grad_norm_(params, 1.0)`

**Config:** epochs=500, batch_size=100, device=H100 80GB

---

## 8. PRE Score (Physics-Residual Error)

5 phương trình MHD residual, tính trên physical-space predictions (FD stencils từ ConvOps_2d):

| # | Residual | Ý nghĩa |
|---|---------|---------|
| R1 | `ω_t + (u·∇)ω − (B·∇)j − ν∇²ω` | Vorticity PDE |
| R2 | `A_t + (u·∇)A − η∇²A` | Induction PDE |
| R3 | `ω − (∂v/∂x − ∂u/∂y)` | Curl consistency |
| R4 | `∂u/∂x + ∂v/∂y` | Div-free velocity |
| R5 | `∂Bx/∂x + ∂By/∂y` | Div-free B field |

```python
score = (|R1| + |R2| + |R3| + |R4| + |R5|).mean(over space & time)
```

**Normalized PRE** (bật với `--normalize-pre`):
```
score_norm = score / (‖ω_pred‖ + ‖A_pred‖ + ε)
```

---

## 9. PRRS Calibration

**Conformal Prediction (CP):** `q̂ = quantile(scores_cal, ⌈(n+1)(1−α)⌉/n)`

**PRRS (τ*):** Tìm threshold τ tối thiểu hóa selective risk trong khi đảm bảo coverage ≥ κ₀ = 0.90:
```
τ* = argmin_τ  E[L | score ≤ τ]    s.t.  P(score ≤ τ) ≥ κ₀
```

---

## 10. Kết quả Seed 0 (Hiện tại)

```
L2_val      = 0.204  (± 0.108)
L2_ood      = NaN

ρ_cal (raw)  = −0.087   [GATE FAIL: cần > 0.3]
ρ_val (raw)  = +0.128
ρ_ood        = NaN

coverage_val = 0.85   [target: 0.90]
coverage_ood = 0.00

PRRS risk    = 0.199
PRE-CP risk  = 0.194
Rand risk    = 0.203
PRRS↓ vs Rand: 2.04%

Training: 46 min / seed  (5.52s/iter)
```

---

## 11. Vấn đề Đang Gặp

### Vấn đề 1: OOD L2 = NaN

**Triệu chứng:** `err_ood.mean() = NaN`, `coverage_ood = 0.00`

**Root cause:**
FNO model output NaN trên OOD inputs trong AR rollout. OOD amplitudes (av ∈ [1.5, 2.0]) cho vorticity tối đa ~25, trong khi training chỉ thấy ~18.8 (av ≤ 1.5). Sau chuẩn hóa với training stats, OOD values ở ~1.3× ngoài range [-1,1]. Qua 5 AR steps, sai số tích lũy → overflow → NaN.

**Fix đã áp dụng:** `torch.nan_to_num(model(inp), nan=0.0)` trong `predict_ar` — ngăn cascade NaN.

**Vấn đề còn lại:** Ngay cả sau nan_to_num, OOD predictions có thể vô nghĩa về mặt vật lý vì model không học được OOD dynamics. Cần kiểm tra L2_ood có finite không sau fix này.

**Hướng giải quyết:**
- [ ] Giảm OOD amplitude gap: `av_hi_ood` 2.0 → 1.8, `aB_hi_ood` 1.2 → 1.0
- [ ] Tăng training range để overlap với OOD: `av_hi` 1.5 → 1.8
- [ ] Dùng physics-informed augmentation cho OOD
- [ ] Kiểm tra xem NaN đến từ solver hay từ FNO forward pass

---

### Vấn đề 2: ρ_cal = −0.087 (Gate FAIL)

**Triệu chứng:** PRE score không tương quan (thậm chí âm) với L2 error.

**Root cause (giả thuyết):**
Model FNO với L2=0.20 đang học "mean prediction" — predict một vortex trung bình, tránh rủi ro. Với các trường hợp khó (av cao, dynamics mạnh):
- L2 error **cao** (model không bắt được dynamics)
- PRE score **thấp** (prediction mượt/flat → residuals nhỏ)

→ Tương quan âm: cases khó có PRE thấp nhưng error cao.

**Normalized PRE** (chia cho energy `√(‖ω‖² + ‖A‖²)`) đã được bật với `--normalize-pre`. Cần chạy lại để kiểm tra.

**Hướng giải quyết:**
- [ ] Chờ kết quả với `--normalize-pre` (đã push)
- [ ] Nếu vẫn thấp: xem xét PRE weighting (R1, R2 quan trọng hơn R3-R5)
- [ ] Cải thiện FNO accuracy (L2 còn 0.20 → cần < 0.10) để PRE informative hơn
- [ ] Tăng epochs (500 → 1000) hoặc learning rate warmup
- [ ] Xem xét dùng physical-space normalization trong PRE thay vì predicted energy

---

### Vấn đề 3: Coverage_val = 0.85 < 0.90

**Triệu chứng:** CP quantile `q̂ = 7.5` không đủ bao phủ 90% val samples.

**Root cause:**
PRE scores trên val set có phân phối khác cal set (cal và val đều in-dist nhưng scores phân tán). Đây thường là hệ quả của ρ thấp — nếu PRE không tương quan với error, threshold CP sẽ kém hiệu quả.

**Hướng giải quyết:**
- [ ] Fix ρ trước → coverage tự cải thiện
- [ ] Dùng conformal prediction trực tiếp trên L2 error thay vì PRE score

---

### Vấn đề 4: Training Loss Cao (Final loss = 0.82)

**Triệu chứng:** Sau 500 epochs, relative L2 training loss vẫn = 0.82. Val L2 = 0.20 (thấp hơn vì val set có amplitudes trung bình).

**Root cause:**
Training set có amplitude range rộng (av ∈ [0.5, 1.5]) — các sample av=1.5 khó hơn nhiều. Model chưa hội tụ với 500 epochs.

**Hướng giải quyết:**
- [ ] Tăng epochs: 500 → 1000 hoặc 2000
- [ ] LR schedule: cosine annealing thay vì StepLR
- [ ] Data curriculum: train trên av thấp trước, tăng dần

---

## 12. Tóm Tắt Ưu Tiên Fix

| Ưu tiên | Vấn đề | Fix đề xuất |
|---------|--------|-------------|
| 🔴 Cao | ρ_cal = −0.09 (gate fail) | Cải thiện FNO accuracy + normalized PRE |
| 🔴 Cao | OOD NaN | Giảm OOD amplitude gap hoặc tăng training range |
| 🟡 Trung | Coverage 0.85 < 0.90 | Phụ thuộc vào ρ fix |
| 🟡 Trung | Training chưa hội tụ | Tăng epochs, cosine LR |
| 🟢 Thấp | Training speed 5.52s/it | Đã tối ưu với batch=100, step=4 |

---

## 13. So Sánh Với Baseline

| Giai đoạn | L2_val | ρ_cal | OOD |
|-----------|--------|--------|-----|
| Random Fourier IC (ban đầu) | ~1.0 | - | - |
| OT IC + PerVarNorm (hiện tại, seed 0) | **0.204** | −0.087 | NaN |
| **Mục tiêu** | < 0.10 | > 0.30 | Finite |
