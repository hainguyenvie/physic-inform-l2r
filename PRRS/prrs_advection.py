#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Residual Rejection Score (PRRS) — 1D Advection Equation
================================================================
Self-contained experiment script. No external submodules required.

Equation:   u_t + v * u_x = 0,   x in [0,2], t in [0,0.5]
IC:         u(x,0) = A * exp(-(x-X)^2),  A in [50,200], X in [0.5,1.0]
Solver:     Crank-Nicolson (periodic BCs)
FNO:        1D, modes=8, width=16, 6 Fourier layers, autoregressive
PRE score:  S(sample) = mean |D(u_pred)|  over (t,x)
            where D = D_t + v*(disc*dt/dx)*D_x  (PDE residual operator)

Comparison:
  1. PRE-CP (paper baseline)      — fixed quantile threshold, no rejection
  2. PRRS-grid                    — learn tau* minimising selective L2
  3. Random rejection             — control baseline
Metric:
  - Selective Risk Curve: acceptance rate vs. L2 of accepted predictions
  - AUC-RC: lower is better
  - Coverage verification (should match paper Figure 12)

Benchmark from paper (Tables 3, Wave 2D — closest reference):
  PRE-CP in-dist  L2 = 1.78e-05 ± 4.61e-07   Coverage = 95.52 ± 0.21%
  PRE-CP out-dist L2 = 2.46e-03 ± 1.25e-05   Coverage = 95.39 ± 0.12%
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve as scipy_solve
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
from functools import reduce
import operator

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print(f"Device: {device}")

# ═════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
cfg = {
    # PDE
    "v": 1.0, "Nx": 200, "Nt": 100, "x_min": 0.0, "x_max": 2.0, "t_end": 0.5,
    # IC sampling
    "A_lo": 50.0, "A_hi": 200.0, "X_lo": 0.5, "X_hi": 1.0,
    # Dataset sizes — increased n_cal for stable CP coverage, more epochs for FNO quality
    "n_train": 100, "n_cal": 500, "n_val": 100,
    # FNO — exact paper config (Appendix G, Table 7)
    "T_in": 1, "T_out": 10, "step": 1, "modes": 8, "width": 16, "num_vars": 1,
    # Training — increased epochs for better convergence (was 100)
    "epochs": 500, "batch_size": 10, "lr": 5e-3,
    "sched_step": 100, "sched_gamma": 0.5,
    # PRE convolution
    "disc": 2,   # subsample every 2nd timestep for PRE
    # PRRS
    "coverage_target": 0.90,   # target acceptance rate
    "n_tau_grid": 300,         # grid resolution for threshold search
    # OOD test
    "A_lo_ood": 200.0, "A_hi_ood": 350.0,
    "X_lo_ood": 0.1, "X_hi_ood": 0.5,
}

# ═════════════════════════════════════════════════════════════════════════════
# 2.  ADVECTION SOLVER (Crank-Nicolson, periodic BCs)
# ═════════════════════════════════════════════════════════════════════════════

def build_cn_matrices(Nx, v, dt, dx):
    """Build Crank-Nicolson advance matrices A (implicit) and B (explicit)."""
    r = v * dt / (4.0 * dx)   # CN coefficient (central diff in space)
    N = Nx
    A = np.eye(N)
    B = np.eye(N)
    for j in range(N):
        jp1 = (j + 1) % N
        jm1 = (j - 1) % N
        # A u^{n+1} = B u^n
        A[j, jp1] += r
        A[j, jm1] -= r
        B[j, jp1] -= r
        B[j, jm1] += r
    return A, B


def solve_advection(amp, xc, cfg):
    """Solve 1D advection and return solution array (Nt+1, Nx)."""
    v, Nx, Nt = cfg["v"], cfg["Nx"], cfg["Nt"]
    x_min, x_max, t_end = cfg["x_min"], cfg["x_max"], cfg["t_end"]
    dx = (x_max - x_min) / Nx
    dt = t_end / Nt
    x = np.linspace(x_min, x_max, Nx, endpoint=False)   # periodic: Nx points
    u = amp * np.exp(-(x - xc) ** 2)

    A, B = build_cn_matrices(Nx, v, dt, dx)
    sols = [u.copy()]
    for _ in range(Nt):
        rhs = B @ u
        u = scipy_solve(A, rhs, assume_a='gen')
        sols.append(u.copy())
    return x, np.array(sols)   # (Nt+1, Nx)


def lhs_sample(n, lo, hi):
    """Latin Hypercube sampling for 2D parameter space."""
    sampler = LatinHypercube(d=2, seed=np.random.randint(0, 10000))
    unit = sampler.random(n)
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    return lo + unit * (hi - lo)   # (n, 2)


def generate_dataset(n, cfg, ood=False):
    """Generate n simulations. Returns u tensor (n, 1, Nx, Nt+1)."""
    if ood:
        params = lhs_sample(n,
                            [cfg["A_lo_ood"], cfg["X_lo_ood"]],
                            [cfg["A_hi_ood"], cfg["X_hi_ood"]])
    else:
        params = lhs_sample(n,
                            [cfg["A_lo"], cfg["X_lo"]],
                            [cfg["A_hi"], cfg["X_hi"]])
    u_list = []
    for amp, xc in tqdm(params, desc="Simulating", leave=False):
        _, u = solve_advection(amp, xc, cfg)
        u_list.append(u)                               # (Nt+1, Nx)
    u_arr = np.array(u_list)                           # (n, Nt+1, Nx)
    u_t = torch.tensor(u_arr, dtype=torch.float32)
    u_t = u_t.permute(0, 2, 1).unsqueeze(1)           # (n, 1, Nx, Nt+1)
    return u_t


def make_loaders(u_data, cfg, shuffle=True):
    """Split into (input=IC, target=rollout) and create DataLoader."""
    T_in, T_out = cfg["T_in"], cfg["T_out"]
    a = u_data[:, :, :, :T_in]                        # (n, 1, Nx, T_in)
    u = u_data[:, :, :, T_in: T_in + T_out]           # (n, 1, Nx, T_out)
    ds = torch.utils.data.TensorDataset(a, u)
    return torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FNO — 1D, multi-variable  (from Other_UQ/Bayesian_Models/Base_FNO.py)
# ═════════════════════════════════════════════════════════════════════════════

class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, n_vars, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / in_ch
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, n_vars, modes, dtype=torch.cfloat))

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, x_ft.shape[1], x_ft.shape[2],
                             x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes] = torch.einsum(
            "bivx,iovx->bovx", x_ft[:, :, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNOBlock1d(nn.Module):
    def __init__(self, modes, n_vars, width):
        super().__init__()
        self.spec  = SpectralConv1d(width, width, n_vars, modes)
        self.mlp1  = nn.Conv2d(width, width, 1)
        self.mlp2  = nn.Conv2d(width, width, 1)
        self.w     = nn.Conv2d(width, width, 1)
        self.b     = nn.Conv2d(1, width, 1)

    def forward(self, x, grid):
        x1 = self.mlp2(F.gelu(self.mlp1(self.spec(x))))
        x2 = self.w(x)
        x3 = self.b(grid)
        return F.gelu(x1 + x2 + x3)


class FNO1D(nn.Module):
    """1D FNO for autoregressive PDE rollout — matches paper Appendix G, Table 7."""
    def __init__(self, T_in, step, modes, n_vars, width):
        super().__init__()
        self.T_in  = T_in
        self.step  = step
        self.n_vars = n_vars

        self.lift = nn.Linear(T_in + 1, width)     # +1 for spatial grid
        self.f0   = FNOBlock1d(modes, n_vars, width)
        self.f1   = FNOBlock1d(modes, n_vars, width)
        self.f2   = FNOBlock1d(modes, n_vars, width)
        self.f3   = FNOBlock1d(modes, n_vars, width)
        self.f4   = FNOBlock1d(modes, n_vars, width)
        self.f5   = FNOBlock1d(modes, n_vars, width)
        self.proj1 = nn.Linear(width, 256)
        self.proj2 = nn.Linear(256, step)

    def get_grid(self, shape, dev):
        bs, nv, nx = shape[0], shape[1], shape[2]
        g = torch.linspace(0, 1, nx).reshape(1, nx, 1).repeat(bs, nv, 1, 1)
        return g.to(dev)

    def forward(self, x):
        # x: (BS, n_vars, Nx, T_in)
        g = self.get_grid(x.shape, x.device)           # (BS, n_vars, Nx, 1)
        x = torch.cat([x, g], dim=-1)                  # (BS, n_vars, Nx, T_in+1)
        x = self.lift(x)                               # (BS, n_vars, Nx, width)
        x = x.permute(0, 3, 1, 2)                     # (BS, width, n_vars, Nx)
        g = g.permute(0, 3, 1, 2)                     # (BS, 1, n_vars, Nx)

        x0 = self.f0(x, g)
        x  = self.f1(x0, g)
        x  = self.f2(x, g) + x0
        x1 = self.f3(x, g)
        x  = self.f4(x1, g)
        x  = self.f5(x, g) + x1

        x = x.permute(0, 2, 3, 1)                     # (BS, n_vars, Nx, width)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)                              # (BS, n_vars, Nx, step)
        return x

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def lp_loss(pred, true, p=2):
    """Relative Lp loss (used in paper via LP-loss)."""
    diff = torch.norm(pred - true, p=p, dim=(-1, -2))
    norm = torch.norm(true, p=p, dim=(-1, -2)) + 1e-8
    return (diff / norm).mean()


def train_one_epoch(model, loader, optimizer, cfg):
    model.train()
    total_loss = 0.0
    T_out, step = cfg["T_out"], cfg["step"]
    for a_batch, u_batch in loader:
        a_batch = a_batch.to(device)
        u_batch = u_batch.to(device)
        optimizer.zero_grad()
        loss = 0.0
        inp = a_batch                                  # (BS, 1, Nx, T_in)
        for t in range(0, T_out, step):
            out = model(inp)                           # (BS, 1, Nx, step)
            target = u_batch[:, :, :, t: t + step]
            loss += lp_loss(out, target)
            inp = torch.cat([inp[:, :, :, step:], out], dim=-1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_ar(model, a, cfg):
    """Autoregressive rollout. Returns predictions (N, 1, Nx, T_out)."""
    model.eval()
    T_out, step = cfg["T_out"], cfg["step"]
    a = a.to(device)
    inp = a
    preds = []
    for t in range(0, T_out, step):
        out = model(inp)
        preds.append(out)
        inp = torch.cat([inp[:, :, :, step:], out], dim=-1)
    return torch.cat(preds, dim=-1).cpu()              # (N, 1, Nx, T_out)


def relative_l2(pred, true):
    """Per-sample relative L2 error. Returns (N,) tensor."""
    diff = (pred - true).reshape(pred.shape[0], -1)
    norm = true.reshape(true.shape[0], -1)
    return (diff.norm(2, dim=1) / (norm.norm(2, dim=1) + 1e-8))


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PRE CONVOLUTION OPERATOR  (mirrors Utils/ConvOps_1d.py)
# ═════════════════════════════════════════════════════════════════════════════

def build_pre_operator(cfg):
    """
    Builds composite PDE residual kernel:
        D = D_t + (v * disc * dt/dx) * D_x
    Stencils act on (BS, Nt, Nx) tensors via 2D convolution.
    """
    v    = cfg["v"]
    disc = cfg["disc"]
    Nx   = cfg["Nx"]
    Nt   = cfg["Nt"]
    dx   = (cfg["x_max"] - cfg["x_min"]) / Nx
    dt   = cfg["t_end"] / Nt

    # D_t stencil: temporal forward difference (time axis = dim 0 of kernel)
    k_t = torch.tensor([[0., -1., 0.],
                         [0.,  0., 0.],
                         [0.,  1., 0.]], dtype=torch.float32)
    # D_x stencil: spatial central difference (space axis = dim 1 of kernel)
    k_x = torch.tensor([[0.,  0., 0.],
                         [-1., 0., 1.],
                         [0.,  0., 0.]], dtype=torch.float32)

    coeff = v * disc * dt / dx
    kernel = k_t + coeff * k_x                        # (3, 3)
    return kernel


def compute_pre(u, kernel):
    """
    Compute PDE residual via 2D convolution.
    u:      (N, Nt, Nx) — subsampled spatio-temporal solution
    kernel: (3, 3)
    Returns: (N, Nt-2, Nx-2) — trimmed to remove boundary artefacts
    """
    u_4d = u.unsqueeze(1)                             # (N, 1, Nt, Nx)
    k = kernel.unsqueeze(0).unsqueeze(0)              # (1, 1, 3, 3)
    res = F.conv2d(u_4d, k, padding=1).squeeze(1)    # (N, Nt, Nx)
    return res[:, 1:-1, 1:-1]                         # trim edges


def pre_score_per_sample(u_pred, kernel, disc):
    """
    PRE score aggregated per sample:  S(i) = mean |D(u_pred_i)|
    u_pred: (N, 1, Nx, T_out)
    Returns: (N,) numpy array
    """
    # reshape to (N, T_out, Nx), subsample time
    u = u_pred[:, 0, :, :].permute(0, 2, 1)          # (N, T_out, Nx)
    u = u[:, ::disc, :]                               # subsample in time
    pre = compute_pre(u, kernel)                      # (N, T_out//disc-2, Nx-2)
    return pre.abs().mean(dim=(1, 2)).numpy()         # (N,)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  CONFORMAL PREDICTION UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def cp_quantile(scores, alpha):
    """
    Inductive CP quantile: q̂_α = ⌈(n+1)(1-α)⌉/n -th quantile of scores.
    scores: (n,) numpy array of non-conformity scores
    """
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return np.quantile(scores, level)


def empirical_coverage(scores_test, qhat):
    """Fraction of test samples with score ≤ q̂ (i.e., within prediction set)."""
    return np.mean(scores_test <= qhat)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  PRRS — LEARNABLE REJECTION THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════

def prrs_calibrate(scores_cal, errors_cal, coverage_target, n_grid=300):
    """
    Grid search for optimal rejection threshold τ*.
    Objective: minimise selective risk (mean L2 of accepted samples)
               subject to acceptance rate ≥ coverage_target.

    Args:
        scores_cal:     (n_cal,)  PRE scores on calibration set
        errors_cal:     (n_cal,)  relative L2 errors on calibration set
        coverage_target: float   minimum acceptance rate
        n_grid:          int     number of threshold candidates
    Returns:
        tau_star: float  — optimal threshold
        results:  dict   — grid search trace for plotting
    """
    tau_grid = np.linspace(scores_cal.min(), scores_cal.max(), n_grid)
    selective_risks = []
    coverages = []

    for tau in tau_grid:
        mask = scores_cal <= tau
        cov = mask.mean()
        coverages.append(cov)
        if mask.sum() >= 3:
            selective_risks.append(errors_cal[mask].mean())
        else:
            selective_risks.append(np.inf)

    selective_risks = np.array(selective_risks)
    coverages = np.array(coverages)

    # Feasible set: acceptance rate >= coverage_target
    feasible = coverages >= coverage_target
    if feasible.sum() == 0:
        # Relax: pick tau with highest acceptance rate
        tau_star = tau_grid[np.argmax(coverages)]
    else:
        # Among feasible, pick lowest selective risk
        feasible_risks = np.where(feasible, selective_risks, np.inf)
        tau_star = tau_grid[np.argmin(feasible_risks)]

    return tau_star, {
        "tau_grid": tau_grid,
        "coverages": coverages,
        "selective_risks": selective_risks,
    }


def risk_coverage_curve(scores, errors, n_thresholds=300):
    """
    Compute Risk-Coverage curve.
    Returns:
        coverages: (K,)   acceptance rates, descending
        risks:     (K,)   selective L2 at each acceptance level
    """
    taus = np.linspace(scores.max(), scores.min(), n_thresholds)
    coverages, risks = [], []
    for tau in taus:
        mask = scores <= tau
        cov = mask.mean()
        if mask.sum() < 2:
            continue
        coverages.append(cov)
        risks.append(errors[mask].mean())
    return np.array(coverages), np.array(risks)


def auc_rc(coverages, risks):
    """Area Under Risk-Coverage curve (lower = better rejector)."""
    # sort by coverage ascending
    idx = np.argsort(coverages)
    return float(np.trapz(risks[idx], coverages[idx]))


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MC DROPOUT REJECTION BASELINE
# ═════════════════════════════════════════════════════════════════════════════

class FNO1D_Dropout(FNO1D):
    """FNO with dropout for MC uncertainty estimation."""
    def __init__(self, *args, dropout_p=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        g = self.get_grid(x.shape, x.device)
        x = torch.cat([x, g], dim=-1)
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)
        g = g.permute(0, 3, 1, 2)

        x0 = self.drop(self.f0(x, g))
        x  = self.drop(self.f1(x0, g))
        x  = self.drop(self.f2(x, g)) + x0
        x1 = self.drop(self.f3(x, g))
        x  = self.drop(self.f4(x1, g))
        x  = self.drop(self.f5(x, g)) + x1

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)


@torch.no_grad()
def mc_dropout_variance(model, a, cfg, K=20):
    """
    Compute predictive variance via K stochastic forward passes (MC Dropout).
    Returns per-sample variance score: (N,)
    """
    model.train()   # keep dropout active
    a = a.to(device)
    T_out, step = cfg["T_out"], cfg["step"]
    preds = []
    for _ in range(K):
        inp = a
        pred_list = []
        for t in range(0, T_out, step):
            out = model(inp)
            pred_list.append(out)
            inp = torch.cat([inp[:, :, :, step:], out], dim=-1)
        preds.append(torch.cat(pred_list, dim=-1).cpu())

    preds = torch.stack(preds, dim=0)                 # (K, N, 1, Nx, T_out)
    var = preds.var(dim=0)                            # (N, 1, Nx, T_out)
    return var.mean(dim=(1, 2, 3)).numpy()             # (N,)


# ═════════════════════════════════════════════════════════════════════════════
# 9.  MAIN EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("PRRS Experiment — 1D Advection Equation")
    print("=" * 65)

    # ── 9.1  Data generation ─────────────────────────────────────────────────
    print("\n[1/7] Generating datasets …")
    u_train = generate_dataset(cfg["n_train"], cfg)
    u_cal   = generate_dataset(cfg["n_cal"],   cfg)
    u_val   = generate_dataset(cfg["n_val"],   cfg)
    u_ood   = generate_dataset(cfg["n_val"],   cfg, ood=True)
    print(f"  train {tuple(u_train.shape)}, cal {tuple(u_cal.shape)}, "
          f"val {tuple(u_val.shape)}, ood {tuple(u_ood.shape)}")

    train_loader = make_loaders(u_train, cfg, shuffle=True)
    a_cal, u_cal_out = u_cal[:, :, :, :cfg["T_in"]], u_cal[:, :, :, cfg["T_in"]:cfg["T_in"]+cfg["T_out"]]
    a_val, u_val_out = u_val[:, :, :, :cfg["T_in"]], u_val[:, :, :, cfg["T_in"]:cfg["T_in"]+cfg["T_out"]]
    a_ood, u_ood_out = u_ood[:, :, :, :cfg["T_in"]], u_ood[:, :, :, cfg["T_in"]:cfg["T_in"]+cfg["T_out"]]

    # ── 9.2  Train FNO ───────────────────────────────────────────────────────
    print("\n[2/7] Training FNO …")
    model = FNO1D(cfg["T_in"], cfg["step"], cfg["modes"],
                  cfg["num_vars"], cfg["width"]).to(device)
    print(f"  FNO params: {model.count_params():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["sched_step"], gamma=cfg["sched_gamma"])

    t0 = time.time()
    train_losses = []
    for ep in tqdm(range(cfg["epochs"]), desc="Training"):
        loss = train_one_epoch(model, train_loader, optimizer, cfg)
        train_losses.append(loss)
        scheduler.step()
    print(f"  Train time: {time.time()-t0:.1f}s  Final loss: {train_losses[-1]:.4e}")

    # ── 9.3  Validate FNO ────────────────────────────────────────────────────
    print("\n[3/7] Evaluating FNO …")
    pred_val = predict_ar(model, a_val, cfg)
    pred_cal = predict_ar(model, a_cal, cfg)
    pred_ood = predict_ar(model, a_ood, cfg)

    err_val = relative_l2(pred_val, u_val_out)
    err_ood = relative_l2(pred_ood, u_ood_out)
    err_cal = relative_l2(pred_cal, u_cal_out)
    print(f"  In-dist  L2 (mean ± std): {err_val.mean():.3e} ± {err_val.std():.3e}")
    print(f"  OOD      L2 (mean ± std): {err_ood.mean():.3e} ± {err_ood.std():.3e}")

    # ── 9.4  PRE scores ───────────────────────────────────────────────────────
    print("\n[4/7] Computing PRE scores …")
    kernel = build_pre_operator(cfg)

    scores_cal = pre_score_per_sample(pred_cal, kernel, cfg["disc"])
    scores_val = pre_score_per_sample(pred_val, kernel, cfg["disc"])
    scores_ood = pre_score_per_sample(pred_ood, kernel, cfg["disc"])
    print(f"  PRE cal  — mean: {scores_cal.mean():.3e}  std: {scores_cal.std():.3e}")
    print(f"  PRE val  — mean: {scores_val.mean():.3e}  std: {scores_val.std():.3e}")
    print(f"  PRE ood  — mean: {scores_ood.mean():.3e}  std: {scores_ood.std():.3e}")

    # ── 9.5  PRE-CP calibration (paper baseline) ─────────────────────────────
    print("\n[5/7] PRE-CP calibration …")
    alpha = 1.0 - cfg["coverage_target"]
    qhat  = cp_quantile(scores_cal, alpha)

    cov_val = empirical_coverage(scores_val, qhat)
    cov_ood = empirical_coverage(scores_ood, qhat)
    print(f"  PRE-CP q̂_α = {qhat:.4e}")
    print(f"  Coverage in-dist: {100*cov_val:.2f}%  (target {100*(1-alpha):.0f}%)")
    print(f"  Coverage OOD    : {100*cov_ood:.2f}%")
    print(f"  >> Paper (Wave): in-dist 95.52±0.21%, OOD 95.39±0.12%")

    # ── 9.6  PRRS calibration ────────────────────────────────────────────────
    print("\n[6/7] PRRS calibration (grid search) …")
    tau_star, trace = prrs_calibrate(
        scores_cal, err_cal.numpy(),
        coverage_target=cfg["coverage_target"],
        n_grid=cfg["n_tau_grid"],
    )
    print(f"  PRRS τ* = {tau_star:.4e}")

    # ── 9.7  Evaluation & comparison ─────────────────────────────────────────
    print("\n[7/7] Comparing methods …")

    # — Risk-Coverage curves —
    rc_pre_cp_val = risk_coverage_curve(scores_val, err_val.numpy())
    rc_pre_cp_ood = risk_coverage_curve(scores_ood, err_ood.numpy())

    # PRRS uses the same score, just a different (learned) threshold point
    # We also compute the full curve to show PRRS is on the SAME curve but at τ*
    prrs_mask_val = scores_val <= tau_star
    prrs_mask_ood = scores_ood <= tau_star

    prrs_cov_val = prrs_mask_val.mean()
    prrs_cov_ood = prrs_mask_ood.mean()
    prrs_risk_val = err_val.numpy()[prrs_mask_val].mean() if prrs_mask_val.sum() > 0 else np.nan
    prrs_risk_ood = err_ood.numpy()[prrs_mask_ood].mean() if prrs_mask_ood.sum() > 0 else np.nan

    # PRE-CP fixed threshold (accept-all baseline: tau = q̂_α from CP)
    precp_mask_val = scores_val <= qhat
    precp_mask_ood = scores_ood <= qhat
    precp_cov_val = precp_mask_val.mean()
    precp_cov_ood = precp_mask_ood.mean()
    precp_risk_val = err_val.numpy()[precp_mask_val].mean() if precp_mask_val.sum() > 0 else np.nan
    precp_risk_ood = err_ood.numpy()[precp_mask_ood].mean() if precp_mask_ood.sum() > 0 else np.nan

    # Random baseline (same acceptance rate as PRRS)
    rand_risk_val_list, rand_risk_ood_list = [], []
    np.random.seed(0)
    for _ in range(50):
        idx_r = np.random.choice(len(err_val), size=prrs_mask_val.sum(), replace=False)
        rand_risk_val_list.append(err_val.numpy()[idx_r].mean())
        idx_r = np.random.choice(len(err_ood), size=max(1, prrs_mask_ood.sum()), replace=False)
        rand_risk_ood_list.append(err_ood.numpy()[idx_r].mean())
    rand_risk_val = np.mean(rand_risk_val_list)
    rand_risk_ood = np.mean(rand_risk_ood_list)

    auc_val = auc_rc(*rc_pre_cp_val)
    auc_ood = auc_rc(*rc_pre_cp_ood)

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n{'Method':<25} {'Accept%':>8} {'Risk(L2)':>12}  {'AUC-RC':>10}")
    print("-" * 60)
    print(f"{'[In-distribution]'}")
    print(f"  {'PRE-CP (accept all)':<23} {'100.0':>8}% {err_val.mean().item():>12.3e}  {'N/A':>10}")
    print(f"  {'PRE-CP (τ=q̂α)':<23} {100*precp_cov_val:>8.1f}% {precp_risk_val:>12.3e}  {auc_val:>10.4f}")
    print(f"  {'PRRS (τ*)':<23} {100*prrs_cov_val:>8.1f}% {prrs_risk_val:>12.3e}  {'(same curve)':>10}")
    print(f"  {'Random (same n)':<23} {100*prrs_cov_val:>8.1f}% {rand_risk_val:>12.3e}  {'N/A':>10}")
    print()
    print(f"{'[Out-of-distribution]'}")
    print(f"  {'PRE-CP (accept all)':<23} {'100.0':>8}% {err_ood.mean().item():>12.3e}  {'N/A':>10}")
    print(f"  {'PRE-CP (τ=q̂α)':<23} {100*precp_cov_ood:>8.1f}% {precp_risk_ood:>12.3e}  {auc_ood:>10.4f}")
    print(f"  {'PRRS (τ*)':<23} {100*prrs_cov_ood:>8.1f}% {prrs_risk_ood:>12.3e}  {'(same curve)':>10}")
    print(f"  {'Random (same n)':<23} {100*prrs_cov_ood:>8.1f}% {rand_risk_ood:>12.3e}  {'N/A':>10}")
    print()
    print(f"Coverage (PRE-CP, α={alpha:.2f}):")
    print(f"  In-dist: {100*cov_val:.2f}%   OOD: {100*cov_ood:.2f}%")
    print(f"  Paper (Wave, in-dist): 95.52±0.21%   OOD: 95.39±0.12%")
    print("=" * 65)

    # ═══════════════════════════════════════════════════════════════════════
    # 10.  PLOTS
    # ═══════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PRRS vs PRE-CP — 1D Advection Equation", fontsize=13)

    # ── Plot 1: Risk-Coverage curve (in-distribution) ────────────────────────
    ax = axes[0]
    cov_rc, risk_rc = rc_pre_cp_val
    ax.plot(cov_rc, risk_rc, 'b-', lw=2, label=f'PRE score curve (AUC={auc_val:.4f})')
    ax.scatter([precp_cov_val], [precp_risk_val], s=120, marker='s', c='orange',
               zorder=5, label=f'PRE-CP τ=q̂α (acc={100*precp_cov_val:.0f}%)')
    ax.scatter([prrs_cov_val], [prrs_risk_val], s=160, marker='*', c='red',
               zorder=6, label=f'PRRS τ* (acc={100*prrs_cov_val:.0f}%)')
    ax.scatter([prrs_cov_val], [rand_risk_val], s=120, marker='D', c='gray',
               zorder=5, label=f'Random (same acc)')
    ax.set_xlabel('Acceptance Rate')
    ax.set_ylabel('Selective L2 Error (accepted)')
    ax.set_title('Risk-Coverage Curve (in-dist)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Risk-Coverage curve (OOD) ────────────────────────────────────
    ax = axes[1]
    cov_rc_ood, risk_rc_ood = rc_pre_cp_ood
    ax.plot(cov_rc_ood, risk_rc_ood, 'g-', lw=2, label=f'PRE score curve (AUC={auc_ood:.4f})')
    ax.scatter([precp_cov_ood], [precp_risk_ood], s=120, marker='s', c='orange',
               zorder=5, label=f'PRE-CP τ=q̂α (acc={100*precp_cov_ood:.0f}%)')
    ax.scatter([prrs_cov_ood], [prrs_risk_ood], s=160, marker='*', c='red',
               zorder=6, label=f'PRRS τ* (acc={100*prrs_cov_ood:.0f}%)')
    ax.scatter([prrs_cov_ood], [rand_risk_ood], s=120, marker='D', c='gray',
               zorder=5, label=f'Random (same acc)')
    ax.set_xlabel('Acceptance Rate')
    ax.set_ylabel('Selective L2 Error (accepted)')
    ax.set_title('Risk-Coverage Curve (OOD)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: PRE-CP coverage validation ────────────────────────────────────
    ax = axes[2]
    alpha_levels = np.arange(0.05, 0.95, 0.05)
    emp_covs = []
    for a_ in alpha_levels:
        q_ = cp_quantile(scores_cal, a_)
        emp_covs.append(empirical_coverage(scores_val, q_))
    ax.plot([0, 1], [0, 1], 'k-', lw=2, label='Ideal')
    ax.plot(1 - alpha_levels, emp_covs, 'b--o', lw=2, ms=5, label='PRE-CP (in-dist)')
    # ood
    emp_covs_ood = []
    for a_ in alpha_levels:
        q_ = cp_quantile(scores_cal, a_)
        emp_covs_ood.append(empirical_coverage(scores_ood, q_))
    ax.plot(1 - alpha_levels, emp_covs_ood, 'g--s', lw=2, ms=5, label='PRE-CP (OOD)')
    ax.set_xlabel('1 − α  (target coverage)')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Coverage Guarantee\n(cf. paper Fig. 3 / Fig. 12)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS, 'prrs_advection_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out_path}")

    # ── Extra plot: PRRS calibration trace ───────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("PRRS Grid-Search Calibration (on cal set)", fontsize=12)

    ax = axes2[0]
    feasible = trace["coverages"] >= cfg["coverage_target"]
    ax.plot(trace["tau_grid"], trace["selective_risks"], 'b-', lw=2, label='Selective Risk(τ)')
    ax.axvline(tau_star, c='red', ls='--', lw=2, label=f'τ* = {tau_star:.3e}')
    ax.axvline(qhat,    c='orange', ls='--', lw=2, label=f'PRE-CP q̂ = {qhat:.3e}')
    ax.fill_between(trace["tau_grid"], 0,
                    trace["selective_risks"].max() * 1.1,
                    where=feasible, alpha=0.1, color='green', label='Feasible (acc≥target)')
    ax.set_xlabel('Threshold τ')
    ax.set_ylabel('Selective L2 on cal set')
    ax.set_title('Selective Risk vs Threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(trace["tau_grid"], 100 * trace["coverages"], 'g-', lw=2)
    ax.axvline(tau_star, c='red', ls='--', lw=2, label=f'τ* = {tau_star:.3e}')
    ax.axvline(qhat,    c='orange', ls='--', lw=2, label=f'PRE-CP q̂ = {qhat:.3e}')
    ax.axhline(100 * cfg["coverage_target"], c='gray', ls=':', lw=1.5,
               label=f'Target {100*cfg["coverage_target"]:.0f}%')
    ax.set_xlabel('Threshold τ')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Acceptance Rate vs Threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(RESULTS, 'prrs_calibration_trace.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Calibration trace saved → {out_path2}")

    # ── Training loss curve ───────────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.semilogy(train_losses, 'b-', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (LP)')
    plt.title('FNO Training — 1D Advection')
    plt.grid(True, alpha=0.3)
    out_path3 = os.path.join(RESULTS, 'training_loss.png')
    plt.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Training loss saved   → {out_path3}")

    return {
        "err_val_mean": err_val.mean().item(),
        "err_ood_mean": err_ood.mean().item(),
        "precp_coverage_val": float(cov_val),
        "precp_coverage_ood": float(cov_ood),
        "prrs_tau_star": float(tau_star),
        "prrs_risk_val": float(prrs_risk_val) if not np.isnan(prrs_risk_val) else None,
        "prrs_risk_ood": float(prrs_risk_ood) if not np.isnan(prrs_risk_ood) else None,
        "precp_risk_val": float(precp_risk_val),
        "precp_risk_ood": float(precp_risk_ood),
        "rand_risk_val": float(rand_risk_val),
        "rand_risk_ood": float(rand_risk_ood),
        "auc_rc_val": auc_val,
        "auc_rc_ood": auc_ood,
    }


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = main()
    results["seed"] = SEED
    out_path = os.path.join(RESULTS, f'advection_results_seed{SEED}.json')
    def _to_py(v):
        if hasattr(v, 'tolist'): return v.tolist()
        if hasattr(v, 'item'):   return v.item()
        return v
    with open(out_path, 'w') as f:
        json.dump({k: _to_py(v) for k, v in results.items()}, f, indent=2)
    print(f"\nNumerical results saved → {out_path}")
