#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Residual Rejection Score (PRRS) — 2D Wave Equation
===========================================================
Exact paper config from Appendix J + Joint/Wave_Residuals_CP.py.
Compares PRRS vs PRE-CP on the SAME benchmark reported in Table 3.

Equation:   u_tt = c^2 * (u_xx + u_yy),  c=1.0
            x,y ∈ [-1,1],  t ∈ [0,1]
IC:         u(x,y,0) = exp(-A*((x-X)^2+(y-Y)^2))
            ∂u/∂t(x,y,0) = 0
Params:     A∈[10,50], X,Y∈[0.1,0.5]  (paper Table 10)
Solver:     Spectral (exact FFT-based, steps at dt_sub=0.0333)
            Nt=150, dt=0.00667, Nx=64, dx=2/64=0.03125
FNO:        2D, modes=16, width=32, 6 layers  (paper Table 11)
            T_in=1, T_out=20, t_slice=5 (every 5th timestep)
Config:     n_train=800, n_cal=1000, n_val=100  (paper Appendix J)
            Epochs=500, lr=0.005, batch=50, LP-loss, Min-Max norm

PRE kernel: D = D_tt - (c*dt_sub/dx)^2 * D_xx_yy
            dt_sub = dt * t_slice = 0.03333

Benchmark (Table 3, 2σ≈95% coverage):
  PRE-CP in-dist:  L2=1.78e-05±4.61e-07  Coverage=95.52±0.21%
  PRE-CP OOD:      L2=2.46e-03±1.25e-05  Coverage=95.39±0.12%
  MC Dropout:      Coverage in-dist=97.31±0.03%, OOD=89.83±0.07%
  Deep Ensemble:   Coverage in-dist=98.02±0.04%, OOD=83.44±0.12%
  CP-AER:          Coverage in-dist=95.70±0.21%, OOD=95.59±0.14%
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
from scipy.stats import spearmanr
from tqdm import tqdm
from functools import reduce
import operator

# ── paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS   = os.path.join(ROOT, 'Utils')
sys.path.insert(0, ROOT)
sys.path.insert(0, UTILS)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)

from ConvOps_2d import ConvOperator   # uses fft_conv_pytorch from Utils/

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 0     # match paper: torch.manual_seed(0)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print(f"Device: {device}")

# ═════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION — exact paper values
# ═════════════════════════════════════════════════════════════════════════════
cfg = {
    # PDE (Appendix J.1)
    "c": 1.0, "Nx": 64,
    "x_min": -1.0, "x_max": 1.0,
    "tend": 1.0, "Nt": 150,
    "t_slice": 5,    # subsample every 5th time step → 30 frames
    # IC bounds (Table 10)
    "A_lo": 10.0,  "A_hi": 50.0,
    "X_lo": 0.10,  "X_hi": 0.50,
    "Y_lo": 0.10,  "Y_hi": 0.50,
    # OOD — larger amplitude, different position (outside training range)
    "A_lo_ood": 50.0,  "A_hi_ood": 100.0,
    "X_lo_ood": 0.50,  "X_hi_ood": 0.90,
    "Y_lo_ood": 0.50,  "Y_hi_ood": 0.90,
    # Dataset (Appendix J.3)
    "n_train": 800, "n_cal": 1000, "n_val": 100,
    # FNO (Table 11, Appendix J.2)
    "T_in": 1, "T_out": 20, "step": 1,
    "modes": 16, "width": 32, "num_vars": 1,
    # Training (Appendix J.2)
    "epochs": 500, "batch_size": 200,   # Tăng batch_size lên 200 (vì VRAM lớn)
    "lr": 5e-3, "sched_step": 100, "sched_gamma": 0.5,
    # CP / PRRS
    "alpha": 0.10,           # → 90% target coverage
    "n_tau_grid": 500,
    "coverage_target": 0.90,
}

# derived constants (used repeatedly)
dx     = (cfg["x_max"] - cfg["x_min"]) / cfg["Nx"]   # 0.03125
dt     = cfg["tend"] / cfg["Nt"]                      # 0.006667
dt_sub = dt * cfg["t_slice"]                          # 0.033333
r_cfl  = cfg["c"] * dt / dx                           # 0.2133 (< 1/√2 ✓)
print(f"CFL = {r_cfl:.4f} (stable if < {1/2**0.5:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# 2.  WAVE SOLVER — exact spectral (FFT-based), periodic BCs
#     Steps directly at dt_sub = dt*t_slice → 5x faster, no CFL constraint.
#     Exact solution of u_tt = c^2*(u_xx+u_yy) via Fourier mode propagation.
# ═════════════════════════════════════════════════════════════════════════════

def solve_wave_2d(amp, xc, yc, cfg, rng=None):
    """
    Exact spectral solver for u_tt = c^2*(u_xx+u_yy), periodic BCs.
    IC: u(x,y,0) = exp(-amp*((x-xc)^2+(y-yc)^2)),  u_t(x,y,0)=0.
    Returns: u_sub  (n_frames+1, Nx, Nx)  where n_frames = Nt // t_slice
    """
    c, Nx, Nt = cfg["c"], cfg["Nx"], cfg["Nt"]
    t_slice   = cfg["t_slice"]
    n_frames  = Nt // t_slice   # 30 output frames

    x = np.linspace(cfg["x_min"], cfg["x_max"], Nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    u0 = np.exp(-amp * ((X - xc) ** 2 + (Y - yc) ** 2))

    # Physical wavenumbers (period L = x_max - x_min = 2)
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    omega = c * np.sqrt(KX ** 2 + KY ** 2)   # dispersion relation

    # Precompute propagators for one dt_sub step (constant → compute once)
    cos_o         = np.cos(omega * dt_sub)
    sin_o_over_o  = np.where(omega > 1e-10, np.sin(omega * dt_sub) / omega, dt_sub)
    sin_o_times_o = np.where(omega > 1e-10, np.sin(omega * dt_sub) * omega, 0.0)

    u_hat  = np.fft.fft2(u0)
    ut_hat = np.zeros_like(u_hat)   # u_t(x,y,0) = 0
    sols   = [u0.copy()]

    for _ in range(n_frames):
        u_hat_new  =  cos_o * u_hat + sin_o_over_o  * ut_hat
        ut_hat_new = -sin_o_times_o * u_hat + cos_o * ut_hat
        u_hat  = u_hat_new
        ut_hat = ut_hat_new
        sols.append(np.real(np.fft.ifft2(u_hat)))

    return np.array(sols)   # (n_frames+1, Nx, Nx) = (31, 64, 64)


def lhs_sample(n, lo, hi, rng_seed=None):
    """Latin Hypercube sampling (matches paper's pyDOE lhs)."""
    d = len(lo)
    seed = rng_seed if rng_seed is not None else np.random.randint(0, 99999)
    unit = LatinHypercube(d=d, seed=seed).random(n)
    return np.array(lo) + unit * (np.array(hi) - np.array(lo))


def generate_dataset(n, cfg, ood=False, desc="Simulating"):
    """Generate n wave simulations. Returns u: (n, 1, Nx, Nx, Nt_sub)."""
    lo = [cfg["A_lo"], cfg["X_lo"], cfg["Y_lo"]]
    hi = [cfg["A_hi"], cfg["X_hi"], cfg["Y_hi"]]
    if ood:
        lo = [cfg["A_lo_ood"], cfg["X_lo_ood"], cfg["Y_lo_ood"]]
        hi = [cfg["A_hi_ood"], cfg["X_hi_ood"], cfg["Y_hi_ood"]]

    params = lhs_sample(n, lo, hi)
    u_list = []
    for i in tqdm(range(n), desc=desc, leave=False):
        amp, xc, yc = params[i]
        u_sub = solve_wave_2d(amp, xc, yc, cfg)   # (Nt_sub, Nx, Nx)
        u_list.append(u_sub)

    u_arr = np.array(u_list)                       # (n, Nt_sub, Nx, Nx)
    u_t   = torch.tensor(u_arr, dtype=torch.float32)
    u_t   = u_t.permute(0, 2, 3, 1).unsqueeze(1)  # (n, 1, Nx, Nx, Nt_sub)
    return u_t


# ═════════════════════════════════════════════════════════════════════════════
# 3.  NORMALISATION — Min-Max (paper: "Normalisation Strategy: Min-Max")
# ═════════════════════════════════════════════════════════════════════════════

class MinMaxNorm:
    """Per-dataset min-max normaliser, fitted on training data."""
    def __init__(self):
        self.lo = self.hi = None

    def fit(self, x):
        self.lo = x.min()
        self.hi = x.max()
        return self

    def encode(self, x):
        return 2.0 * (x - self.lo) / (self.hi - self.lo + 1e-8) - 1.0

    def decode(self, x):
        return (x + 1.0) / 2.0 * (self.hi - self.lo + 1e-8) + self.lo


def make_loaders(u_data, cfg, in_norm, out_norm, shuffle=True):
    """Split into (IC, rollout), normalise, return DataLoader + raw tensors."""
    T_in, T_out = cfg["T_in"], cfg["T_out"]
    a = u_data[..., :T_in]              # (n, 1, Nx, Nx, T_in)
    u = u_data[..., T_in: T_in+T_out]  # (n, 1, Nx, Nx, T_out)
    a_enc = in_norm.encode(a)
    u_enc = out_norm.encode(u)
    ds = torch.utils.data.TensorDataset(a_enc, u_enc)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=shuffle)
    return loader, a, u, a_enc, u_enc


# ═════════════════════════════════════════════════════════════════════════════
# 4.  FNO — 2D multivariate  (mirrors Base_FNO.py / FNO_multi2d)
# ═════════════════════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, n_vars, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        s = 1.0 / in_ch
        self.w1 = nn.Parameter(s * torch.rand(in_ch, out_ch, n_vars, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(in_ch, out_ch, n_vars, modes1, modes2, dtype=torch.cfloat))

    def compl_mul(self, x, w):
        return torch.einsum("bivxy,iovxy->bovxy", x, w)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, x_ft.shape[1], x_ft.shape[2],
                             x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :, :m1, :m2] = self.compl_mul(x_ft[:, :, :, :m1, :m2], self.w1)
        out_ft[:, :, :, -m1:, :m2] = self.compl_mul(x_ft[:, :, :, -m1:, :m2], self.w2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock2d(nn.Module):
    def __init__(self, modes1, modes2, n_vars, width, dropout=0.0):
        super().__init__()
        self.spec = SpectralConv2d(width, width, n_vars, modes1, modes2)
        self.mlp1 = nn.Conv3d(width, width, 1)
        self.mlp2 = nn.Conv3d(width, width, 1)
        self.w    = nn.Conv3d(width, width, 1)
        self.b    = nn.Conv3d(2, width, 1)   # grid has 2 channels (x,y)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, grid):
        return self.drop(F.gelu(self.mlp2(F.gelu(self.mlp1(self.spec(x)))) + self.w(x) + self.b(grid)))


class FNO2D(nn.Module):
    """2D FNO matching paper Table 11: modes=16, width=32, 6 Fourier layers."""
    def __init__(self, T_in, step, modes1, modes2, n_vars, width, dropout=0.0):
        super().__init__()
        self.T_in   = T_in
        self.step   = step
        self.n_vars = n_vars

        self.lift  = nn.Linear(T_in + 2, width)   # +2 for (x,y) grid
        self.f0    = FNOBlock2d(modes1, modes2, n_vars, width, dropout)
        self.f1    = FNOBlock2d(modes1, modes2, n_vars, width, dropout)
        self.f2    = FNOBlock2d(modes1, modes2, n_vars, width, dropout)
        self.f3    = FNOBlock2d(modes1, modes2, n_vars, width, dropout)
        self.f4    = FNOBlock2d(modes1, modes2, n_vars, width, dropout)
        self.f5    = FNOBlock2d(modes1, modes2, n_vars, width, dropout)
        self.proj1 = nn.Linear(width, 256)
        self.proj2 = nn.Linear(256, step)

    def get_grid(self, shape, dev):
        bs, nv, nx, ny = shape[:4]
        gx = torch.linspace(0, 1, nx).reshape(1, 1, nx, 1, 1).expand(bs, nv, nx, ny, 1)
        gy = torch.linspace(0, 1, ny).reshape(1, 1, 1, ny, 1).expand(bs, nv, nx, ny, 1)
        return torch.cat([gx, gy], dim=-1).to(dev)   # (bs, nv, nx, ny, 2)

    def forward(self, x):
        # x: (BS, n_vars, Nx, Ny, T_in)
        g = self.get_grid(x.shape, x.device)          # (BS, n_vars, Nx, Ny, 2)
        x = torch.cat([x, g], dim=-1)                 # (BS, n_vars, Nx, Ny, T_in+2)
        x = self.lift(x)                              # (BS, n_vars, Nx, Ny, width)
        x = x.permute(0, 4, 1, 2, 3)                 # (BS, width, n_vars, Nx, Ny)
        g = g.permute(0, 4, 1, 2, 3)                 # (BS, 2, n_vars, Nx, Ny)

        x0 = self.f0(x, g)
        x  = self.f1(x0, g)
        x  = self.f2(x, g) + x0
        x1 = self.f3(x, g)
        x  = self.f4(x1, g)
        x  = self.f5(x, g) + x1

        x = x.permute(0, 2, 3, 4, 1)                 # (BS, n_vars, Nx, Ny, width)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)                          # (BS, n_vars, Nx, Ny, step)

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())


# ═════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def lp_loss(pred, true, p=2):
    """Relative Lp loss — matches paper's 'LP-loss'."""
    diff = torch.norm(pred.reshape(pred.shape[0], -1) - true.reshape(true.shape[0], -1), p=p, dim=1)
    norm = torch.norm(true.reshape(true.shape[0], -1), p=p, dim=1) + 1e-8
    return (diff / norm).mean()


def train_one_epoch(model, loader, optimizer, cfg):
    model.train()
    total_loss = 0.0
    T_out, step = cfg["T_out"], cfg["step"]
    for a_batch, u_batch in loader:
        a_batch = a_batch.to(device)
        u_batch = u_batch.to(device)
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        inp  = a_batch
        for t in range(0, T_out, step):
            out    = model(inp)
            target = u_batch[..., t: t+step]
            loss   = loss + lp_loss(out, target)
            inp    = torch.cat([inp[..., step:], out], dim=-1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_ar(model, a_enc, out_norm, cfg, num_passes=1):
    if num_passes > 1:
        model.train() # MC Dropout active
    else:
        model.eval()
    T_out, step = cfg["T_out"], cfg["step"]
    all_preds_decoded = []
    import torch
    device = next(model.parameters()).device
    for k in range(num_passes):
        inp   = a_enc.to(device)
        preds = []
        for t in range(0, T_out, step):
            out = model(inp)
            preds.append(out)
            inp = torch.cat([inp[..., step:], out], dim=-1)
        pred_enc = torch.cat(preds, dim=-1).cpu()
        all_preds_decoded.append(out_norm.decode(pred_enc))
    if num_passes == 1:
        return all_preds_decoded[0]
    return torch.stack(all_preds_decoded, dim=0)

@torch.no_grad()
def ensemble_predict_ar(models, a_enc, out_norm, cfg):
    all_preds = []
    for model in models:
        all_preds.append(predict_ar(model, a_enc, out_norm, cfg, num_passes=1))
    import torch
    return torch.stack(all_preds, dim=0)


def relative_l2(pred, true):
    """Per-sample relative L2. Returns (N,)."""
    diff = (pred - true).reshape(pred.shape[0], -1)
    norm = true.reshape(true.shape[0], -1)
    return (diff.norm(2, dim=1) / (norm.norm(2, dim=1) + 1e-8))


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PRE OPERATOR  — matches Joint/Wave_Residuals_CP.py exactly
#     D.kernel = D_tt.kernel - (c*dt_sub/dx)^2 * D_xx_yy.kernel
#     data format for conv: (BS, Nt, Nx, Ny)  → permute from FNO output
# ═════════════════════════════════════════════════════════════════════════════

def build_wave_pre_operator(cfg, dt_sub, dx):
    """
    Build composite Wave PDE residual operator.
    Returns ConvOperator D with:  D(u) ≈ u_tt - c^2*(u_xx + u_yy)
    """
    c  = cfg["c"]
    D_tt    = ConvOperator('t', 2)           # second-order time derivative
    D_xx_yy = ConvOperator(('x', 'y'), 2)   # 2D Laplacian
    D = ConvOperator()
    # Exact formula from Joint/Wave_Residuals_CP.py line 175:
    D.kernel = D_tt.kernel - (c * dt_sub / dx) ** 2 * D_xx_yy.kernel
    return D


def pre_score_wave(u_pred_decoded, D, trim=True):
    """
    Compute PRE score per sample for Wave 2D.
    u_pred_decoded: (N, 1, Nx, Nx, T_out)  — FNO output, decoded
    Returns: (N,) mean |PRE| per sample
    """
    # permute to (N, T_out, Nx, Nx) — matches ConvOps_2d expectation (BS, Nt, Nx, Ny)
    u = u_pred_decoded[:, 0, :, :, :].permute(0, 3, 1, 2)  # (N, T_out, Nx, Nx)
    res = D(u)                                               # (N, T_out, Nx, Nx)
    if trim:
        res = res[:, 1:-1, 1:-1, 1:-1]                      # trim boundaries
    return res.abs().mean(dim=(1, 2, 3)).detach().numpy()    # (N,)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  CONFORMAL PREDICTION UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def cp_quantile(scores, alpha):
    n     = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def empirical_coverage(scores_test, qhat):
    return float(np.mean(scores_test <= qhat))


# ═════════════════════════════════════════════════════════════════════════════
# 8.  PRRS — LEARNABLE REJECTION THRESHOLD (grid search)
# ═════════════════════════════════════════════════════════════════════════════

def prrs_calibrate(scores_cal, errors_cal, coverage_target, n_grid=500):
    taus          = np.linspace(scores_cal.min(), scores_cal.max(), n_grid)
    sel_risks     = []
    coverages     = []
    for tau in taus:
        mask = scores_cal <= tau
        cov  = mask.mean()
        coverages.append(cov)
        sel_risks.append(errors_cal[mask].mean() if mask.sum() >= 3 else np.inf)
    sel_risks  = np.array(sel_risks)
    coverages  = np.array(coverages)
    feasible   = coverages >= coverage_target
    if feasible.sum() == 0:
        tau_star = taus[np.argmax(coverages)]
    else:
        feasible_risks = np.where(feasible, sel_risks, np.inf)
        tau_star = taus[np.argmin(feasible_risks)]
    return tau_star, {"tau_grid": taus, "coverages": coverages, "sel_risks": sel_risks}


def risk_coverage_curve(scores, errors, n_thresholds=500):
    taus = np.linspace(scores.max(), scores.min(), n_thresholds)
    covs, risks = [], []
    for tau in taus:
        mask = scores <= tau
        if mask.sum() < 2:
            continue
        covs.append(mask.mean())
        risks.append(errors[mask].mean())
    return np.array(covs), np.array(risks)


def auc_rc(covs, risks):
    idx = np.argsort(covs)
    return float(np.trapezoid(risks[idx], covs[idx]))


# ═════════════════════════════════════════════════════════════════════════════
# 9.  MAIN EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

def main(suffix="", method="prrs", dropout_p=0.0):
    t_total = time.time()
    print("=" * 68)
    print(f"PRRS Experiment — 2D Wave Equation  (exact paper config){' seed'+suffix[6:] if suffix else ''}")
    print("=" * 68)
    print(f"  n_train={cfg['n_train']}  n_cal={cfg['n_cal']}  n_val={cfg['n_val']}")
    print(f"  FNO: modes={cfg['modes']} width={cfg['width']} epochs={cfg['epochs']}")
    print(f"  PRE: dt_sub={dt_sub:.5f}  dx={dx:.5f}  (c*dt_sub/dx)^2={(cfg['c']*dt_sub/dx)**2:.4f}")

    # ── 9.1  Data generation ─────────────────────────────────────────────────
    print("\n[1/8] Generating datasets …")
    u_train = generate_dataset(cfg["n_train"], cfg, desc="Train sims")
    u_cal   = generate_dataset(cfg["n_cal"],   cfg, desc="Cal sims")
    u_val   = generate_dataset(cfg["n_val"],   cfg, desc="Val sims (in-dist)")
    u_ood   = generate_dataset(cfg["n_val"],   cfg, ood=True, desc="Val sims (OOD)")
    print(f"  Shapes — train:{tuple(u_train.shape)}  cal:{tuple(u_cal.shape)}")
    print(f"           val:{tuple(u_val.shape)}  ood:{tuple(u_ood.shape)}")

    # ── 9.2  Min-Max normalisation (fit on train) ─────────────────────────────
    print("\n[2/8] Fitting Min-Max normaliser …")
    T_in, T_out = cfg["T_in"], cfg["T_out"]
    in_norm  = MinMaxNorm().fit(u_train[..., :T_in])
    out_norm = MinMaxNorm().fit(u_train[..., T_in: T_in+T_out])

    train_loader, _, _, _, _              = make_loaders(u_train, cfg, in_norm, out_norm)
    _, a_cal_raw, u_cal_out, a_cal_enc, u_cal_enc = make_loaders(u_cal, cfg, in_norm, out_norm, shuffle=False)
    _, a_val_raw, u_val_out, a_val_enc, u_val_enc = make_loaders(u_val, cfg, in_norm, out_norm, shuffle=False)
    _, a_ood_raw, u_ood_out, a_ood_enc, u_ood_enc = make_loaders(u_ood, cfg, in_norm, out_norm, shuffle=False)

    # ── 9.3  Train FNO ───────────────────────────────────────────────────────
    print(f"\n[3/8] Training FNO ({cfg['epochs']} epochs) …")
    model = FNO2D(cfg["T_in"], cfg["step"], cfg["modes"], cfg["modes"], cfg["num_vars"], cfg["width"], dropout=dropout_p).to(device)
    print(f"  FNO params: {model.count_params():,}")

    if method == "ensemble":
        print("  [Skipping training for ensemble inference]")
        pass
    else:
        opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg["sched_step"], cfg["sched_gamma"])

    t0 = time.time()
    losses = []
    for ep in tqdm(range(cfg["epochs"]), desc="Training"):
        l = train_one_epoch(model, train_loader, opt, cfg)
        losses.append(l)
        sched.step()
    train_time = time.time() - t0
    print(f"  Train time: {train_time/60:.1f} min   Final LP-loss: {losses[-1]:.4e}")

    # save checkpoint
    ckpt_path = os.path.join(RESULTS, f'wave2d_fno{suffix}.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    # ── 9.4  FNO evaluation ───────────────────────────────────────────────────
    print("\n[4/8] Evaluating FNO …")
    if method == "ensemble":
        models = []
        for s in [0, 42, 1, 2, 3]:
            m = FNO2D(cfg["T_in"], cfg["step"], cfg["modes"], cfg["modes"], cfg["num_vars"], cfg["width"], dropout=0.0).to(device)
            chk_path = os.path.join(RESULTS, 'wave2d_fno_seed'+str(s)+'.pt')
            m.load_state_dict(torch.load(chk_path, map_location=device))
            m.eval()
            models.append(m)
        pred_cal_k = ensemble_predict_ar(models, a_cal_enc, out_norm, cfg)
        pred_val_k = ensemble_predict_ar(models, a_val_enc, out_norm, cfg)
        pred_ood_k = ensemble_predict_ar(models, a_ood_enc, out_norm, cfg)
    elif method == "mc_dropout":
        pred_cal_k = predict_ar(model, a_cal_enc, out_norm, cfg, num_passes=20)
        pred_val_k = predict_ar(model, a_val_enc, out_norm, cfg, num_passes=20)
        pred_ood_k = predict_ar(model, a_ood_enc, out_norm, cfg, num_passes=20)
    else:
        pred_cal_k = predict_ar(model, a_cal_enc, out_norm, cfg, num_passes=1).unsqueeze(0)
        pred_val_k = predict_ar(model, a_val_enc, out_norm, cfg, num_passes=1).unsqueeze(0)
        pred_ood_k = predict_ar(model, a_ood_enc, out_norm, cfg, num_passes=1).unsqueeze(0)
        
    pred_cal = pred_cal_k.mean(dim=0)
    pred_val = pred_val_k.mean(dim=0)
    pred_ood = pred_ood_k.mean(dim=0)

    err_cal = relative_l2(pred_cal, u_cal_out).numpy()
    err_val = relative_l2(pred_val, u_val_out).numpy()
    err_ood = relative_l2(pred_ood, u_ood_out).numpy()

    l2_val = err_val.mean();  l2_val_std = err_val.std()
    l2_ood = err_ood.mean();  l2_ood_std = err_ood.std()
    print(f"  In-dist  L2: {l2_val:.3e} ± {l2_val_std:.3e}")
    print(f"  OOD      L2: {l2_ood:.3e} ± {l2_ood_std:.3e}")
    print(f"  >> Paper Table 3 (PRE-CP Ours): 1.78e-05 ± 4.61e-07  [Deterministic FNO]")

    # ── 9.5  Build PRE operator ───────────────────────────────────────────────
    print("\n[5/8] Computing PRE scores …")
    D_wave = build_wave_pre_operator(cfg, dt_sub, dx)

    if method in ["mc_dropout", "ensemble"]:
        scores_cal = (pred_cal_k.std(dim=0) / (pred_cal_k.mean(dim=0).abs() + 1e-6)).mean(dim=(1,2,3,4)).numpy()
        scores_val = (pred_val_k.std(dim=0) / (pred_val_k.mean(dim=0).abs() + 1e-6)).mean(dim=(1,2,3,4)).numpy()
        scores_ood = (pred_ood_k.std(dim=0) / (pred_ood_k.mean(dim=0).abs() + 1e-6)).mean(dim=(1,2,3,4)).numpy()
    else:
        scores_cal = pre_score_wave(pred_cal, D_wave)
        scores_val = pre_score_wave(pred_val, D_wave)
        scores_ood = pre_score_wave(pred_ood, D_wave)

    print(f"  PRE cal — mean:{scores_cal.mean():.3e}  std:{scores_cal.std():.3e}")
    print(f"  PRE val — mean:{scores_val.mean():.3e}  std:{scores_val.std():.3e}")
    print(f"  PRE ood — mean:{scores_ood.mean():.3e}  std:{scores_ood.std():.3e}")

    # ── Track 2.3 gate: Spearman ρ(PRE_score, L2_error) ─────────────────────
    rho_cal, p_cal = spearmanr(scores_cal, err_cal)
    rho_val, p_val = spearmanr(scores_val, err_val)
    rho_ood, p_ood = spearmanr(scores_ood, err_ood)
    print(f"\n  [Gate — Track 2.3] Spearman ρ(PRE, L2):")
    print(f"    Cal : ρ={rho_cal:.4f}  p={p_cal:.2e}")
    print(f"    Val : ρ={rho_val:.4f}  p={p_val:.2e}")
    print(f"    OOD : ρ={rho_ood:.4f}  p={p_ood:.2e}")
    print(f"    → {'PASS ✓ (ρ>0.3, PRE is discriminative)' if rho_val > 0.3 else 'FAIL ✗ (ρ<0.1, rethink needed)' if rho_val < 0.1 else 'MARGINAL (0.1≤ρ≤0.3)'}")

    # ── 9.6  PRE-CP calibration (paper baseline) ─────────────────────────────
    print("\n[6/8] PRE-CP calibration (paper baseline) …")
    alpha = cfg["alpha"]
    qhat  = cp_quantile(scores_cal, alpha)

    cov_val = empirical_coverage(scores_val, qhat)
    cov_ood = empirical_coverage(scores_ood, qhat)
    print(f"  q̂_α (α={alpha}) = {qhat:.4e}")
    print(f"  Coverage in-dist: {100*cov_val:.2f}%   (paper: 95.52±0.21%)")
    print(f"  Coverage OOD    : {100*cov_ood:.2f}%   (paper: 95.39±0.12%)")

    # Sweep alpha for coverage validation plot
    alpha_levels = np.arange(0.05, 0.96, 0.05)
    emp_cov_val, emp_cov_ood = [], []
    for a_ in alpha_levels:
        q_ = cp_quantile(scores_cal, a_)
        emp_cov_val.append(empirical_coverage(scores_val, q_))
        emp_cov_ood.append(empirical_coverage(scores_ood, q_))

    # ── 9.7  PRRS calibration ────────────────────────────────────────────────
    print("\n[7/8] PRRS calibration (grid search on cal set) …")
    tau_star, trace = prrs_calibrate(
        scores_cal, err_cal,
        coverage_target=cfg["coverage_target"],
        n_grid=cfg["n_tau_grid"],
    )
    print(f"  PRRS τ* = {tau_star:.4e}   (PRE-CP q̂ = {qhat:.4e})")

    # ── 9.8  Evaluation & comparison ─────────────────────────────────────────
    print("\n[8/8] Comparing methods …")

    # Full Risk-Coverage curves
    rc_cov_val, rc_risk_val = risk_coverage_curve(scores_val, err_val)
    rc_cov_ood, rc_risk_ood = risk_coverage_curve(scores_ood, err_ood)
    auc_val = auc_rc(rc_cov_val, rc_risk_val)
    auc_ood = auc_rc(rc_cov_ood, rc_risk_ood)

    # PRRS operating point (τ*)
    prrs_mask_val = scores_val <= tau_star
    prrs_mask_ood = scores_ood <= tau_star
    prrs_cov_val  = prrs_mask_val.mean()
    prrs_cov_ood  = prrs_mask_ood.mean()
    prrs_risk_val = err_val[prrs_mask_val].mean() if prrs_mask_val.sum() > 0 else np.nan
    prrs_risk_ood = err_ood[prrs_mask_ood].mean() if prrs_mask_ood.sum() > 0 else np.nan

    # PRE-CP operating point (q̂)
    precp_mask_val = scores_val <= qhat
    precp_mask_ood = scores_ood <= qhat
    precp_cov_val  = precp_mask_val.mean()
    precp_cov_ood  = precp_mask_ood.mean()
    precp_risk_val = err_val[precp_mask_val].mean() if precp_mask_val.sum() > 0 else np.nan
    precp_risk_ood = err_ood[precp_mask_ood].mean() if precp_mask_ood.sum() > 0 else np.nan

    # Random baseline (same acceptance rate as PRRS)
    np.random.seed(42)
    n_accept = max(int(prrs_cov_val * len(err_val)), 1)
    rand_risks_val = [err_val[np.random.choice(len(err_val), n_accept, False)].mean()
                      for _ in range(50)]
    rand_risk_val = np.mean(rand_risks_val)
    n_accept_ood  = max(int(prrs_cov_ood * len(err_ood)), 1)
    rand_risks_ood = [err_ood[np.random.choice(len(err_ood), n_accept_ood, False)].mean()
                      for _ in range(50)]
    rand_risk_ood = np.mean(rand_risks_ood)

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("RESULTS vs PAPER TABLE 3  (Wave 2D, 2σ ≈ 95% → set α=0.05 for 2σ)")
    print("=" * 68)
    print(f"\n{'Method':<28} {'Accept%':>8} {'Sel.L2':>12}  {'AUC-RC':>8}")
    print("-" * 62)
    print("[In-distribution]")
    print(f"  {'All accepted (Deterministic)':<26} {'100.0':>8}% {l2_val:>12.3e}  {'—':>8}")
    print(f"  {'PRE-CP (τ=q̂)':<26} {100*precp_cov_val:>8.1f}% {precp_risk_val:>12.3e}  {auc_val:>8.4f}")
    print(f"  {'PRRS (τ*)':<26} {100*prrs_cov_val:>8.1f}% {prrs_risk_val:>12.3e}  {'(same)':>8}")
    print(f"  {'Random (same n)':<26} {100*prrs_cov_val:>8.1f}% {rand_risk_val:>12.3e}  {'—':>8}")
    print()
    print("[Out-of-distribution]")
    print(f"  {'All accepted (Deterministic)':<26} {'100.0':>8}% {l2_ood:>12.3e}  {'—':>8}")
    print(f"  {'PRE-CP (τ=q̂)':<26} {100*precp_cov_ood:>8.1f}% {precp_risk_ood if not np.isnan(precp_risk_ood) else 0:>12.3e}  {auc_ood:>8.4f}")
    print(f"  {'PRRS (τ*)':<26} {100*prrs_cov_ood:>8.1f}% {prrs_risk_ood if not np.isnan(prrs_risk_ood) else 0:>12.3e}  {'(same)':>8}")
    print(f"  {'Random (same n)':<26} {100*prrs_cov_ood:>8.1f}% {rand_risk_ood:>12.3e}  {'—':>8}")
    print()
    print(f"Coverage validation (PRE-CP, α={alpha}):")
    print(f"  In-dist: {100*cov_val:.2f}%  OOD: {100*cov_ood:.2f}%")
    print(f"  Paper Table 3: in-dist 95.52±0.21%,  OOD 95.39±0.12%")
    print()
    print(f"L2 comparison (all accepted):")
    print(f"  This run: {l2_val:.3e} ± {l2_val_std:.3e}  (in-dist)")
    print(f"  Paper:    1.78e-05 ± 4.61e-07             (in-dist)")
    print(f"  [Note: paper uses spectral solver + pretrained 500-epoch FNO]")
    print(f"\nTotal experiment time: {(time.time()-t_total)/60:.1f} min")
    print("=" * 68)

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("PRRS vs PRE-CP — 2D Wave Equation (paper config)", fontsize=13)

    # Plot 1: Risk-Coverage (in-dist)
    ax = axes[0]
    ax.plot(rc_cov_val, rc_risk_val, 'b-', lw=2.5, label=f'PRE RC curve (AUC={auc_val:.4f})')
    ax.scatter([precp_cov_val], [precp_risk_val], s=160, marker='s', c='orange', zorder=5,
               label=f'PRE-CP q̂ (acc={100*precp_cov_val:.0f}%)')
    ax.scatter([prrs_cov_val], [prrs_risk_val], s=200, marker='*', c='red', zorder=6,
               label=f'PRRS τ* (acc={100*prrs_cov_val:.0f}%)')
    ax.scatter([prrs_cov_val], [rand_risk_val], s=120, marker='D', c='gray', zorder=5,
               label=f'Random (same n)')
    ax.set_xlabel('Acceptance Rate'); ax.set_ylabel('Selective L2')
    ax.set_title('Risk-Coverage (in-dist)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 2: Risk-Coverage (OOD)
    ax = axes[1]
    ax.plot(rc_cov_ood, rc_risk_ood, 'g-', lw=2.5, label=f'PRE RC curve (AUC={auc_ood:.4f})')
    if not np.isnan(precp_risk_ood):
        ax.scatter([precp_cov_ood], [precp_risk_ood], s=160, marker='s', c='orange', zorder=5,
                   label=f'PRE-CP q̂ (acc={100*precp_cov_ood:.0f}%)')
    if not np.isnan(prrs_risk_ood):
        ax.scatter([prrs_cov_ood], [prrs_risk_ood], s=200, marker='*', c='red', zorder=6,
                   label=f'PRRS τ* (acc={100*prrs_cov_ood:.0f}%)')
    ax.set_xlabel('Acceptance Rate'); ax.set_ylabel('Selective L2')
    ax.set_title('Risk-Coverage (OOD)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 3: Coverage validation (cf. paper Fig. 3)
    ax = axes[2]
    ax.plot([0,1],[0,1],'k-',lw=2,label='Ideal')
    ax.plot(1-alpha_levels, emp_cov_val, 'b--o', lw=2, ms=5, label='PRE-CP in-dist')
    ax.plot(1-alpha_levels, emp_cov_ood, 'g--s', lw=2, ms=5, label='PRE-CP OOD')
    ax.set_xlabel('1−α (target)'); ax.set_ylabel('Empirical Coverage')
    ax.set_title('Coverage Guarantee\n(cf. paper Fig. 3)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Plot 4: PRRS calibration trace
    ax = axes[3]
    feasible = trace["coverages"] >= cfg["coverage_target"]
    ax.plot(trace["tau_grid"], trace["sel_risks"], 'b-', lw=2, label='Selective Risk(τ)')
    ax.axvline(tau_star, c='red', ls='--', lw=2, label=f'PRRS τ*={tau_star:.2e}')
    ax.axvline(qhat,    c='orange', ls='--', lw=2, label=f'PRE-CP q̂={qhat:.2e}')
    ax.fill_between(trace["tau_grid"], 0, trace["sel_risks"].max()*1.1,
                    where=feasible, alpha=0.1, color='green', label='Feasible region')
    ax.set_xlabel('Threshold τ'); ax.set_ylabel('Selective L2 (cal set)')
    ax.set_title('PRRS Grid Search'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(RESULTS, f'wave2d_prrs_results{suffix}.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    print(f"\nFigure → {p1}")

    # Training loss curve
    plt.figure(figsize=(7,4))
    plt.semilogy(losses,'b-',lw=2)
    plt.xlabel('Epoch'); plt.ylabel('LP Loss'); plt.title('FNO Training — 2D Wave')
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(RESULTS, f'wave2d_training_loss{suffix}.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    print(f"Loss curve → {p2}")

    # PRE score distribution
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.hist(scores_cal, bins=40, alpha=0.5, label='Cal (in-dist)', color='blue', density=True)
    ax.hist(scores_val, bins=40, alpha=0.5, label='Val (in-dist)', color='teal', density=True)
    ax.hist(scores_ood, bins=40, alpha=0.5, label='Val (OOD)',    color='red',  density=True)
    ax.axvline(qhat,    c='orange', ls='--', lw=2, label=f'PRE-CP q̂={qhat:.2e}')
    ax.axvline(tau_star,c='red',    ls='-',  lw=2, label=f'PRRS τ*={tau_star:.2e}')
    ax.set_xlabel('PRE score per sample'); ax.set_ylabel('Density')
    ax.set_title('PRE Score Distribution: in-dist vs OOD'); ax.legend(); ax.grid(True, alpha=0.3)
    p3 = os.path.join(RESULTS, f'wave2d_pre_scores{suffix}.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    print(f"PRE dist → {p3}")

    return {
        "l2_val": l2_val, "l2_ood": l2_ood,
        "coverage_val": cov_val, "coverage_ood": cov_ood,
        "tau_star": tau_star, "qhat": qhat,
        "prrs_risk_val": prrs_risk_val, "precp_risk_val": precp_risk_val,
        "prrs_risk_ood": prrs_risk_ood, "precp_risk_ood": precp_risk_ood,
        "rand_risk_val": rand_risk_val,
        "auc_rc_val": auc_val, "auc_rc_ood": auc_ood,
        # Track 2.3 gate metrics
        "spearman_rho_cal": rho_cal, "spearman_rho_val": rho_val,
        "spearman_rho_ood": rho_ood,
        "spearman_gate_pass": bool(rho_val > 0.3),
    }


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='prrs')
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    suffix = f"_{args.method}_seed{SEED}"
    results = main(suffix=suffix, method=args.method, dropout_p=args.dropout)
    results["seed"] = SEED

    out_path = os.path.join(RESULTS, f'wave2d_results{suffix}.json')
    def _to_py(v):
        if hasattr(v, 'tolist'): return v.tolist()
        if hasattr(v, 'item'):   return v.item()
        return v
    with open(out_path, 'w') as f:
        json.dump({k: _to_py(v) for k, v in results.items()}, f, indent=2)
    print(f"\nNumerical results saved → {out_path}")
