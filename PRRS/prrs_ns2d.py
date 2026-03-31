#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Residual Rejection Score (PRRS) — 2D Incompressible Navier-Stokes
==========================================================================
Pseudo-spectral solver (vorticity–stream-function + pressure Poisson).
4 output variables: u, v, p, w  |  PRE = continuity + momentum residuals

PDE:  ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν∇²u
      ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν∇²v
      ∂u/∂x + ∂v/∂y = 0
      ω = ∂v/∂x - ∂u/∂y

IC:   u(x,y,0) = aa·sin(2πy/L),  v(x,y,0) = bb·sin(2πx/L)
Params: aa,bb ∈ [0.5,1.0]  (paper Joint/NS_Residuals_CP.py)

FNO:  2D multi-var, modes=8, width=16, n_vars=4
      T_in=1, T_out=20, epochs=500, batch=5  (paper config)
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

from ConvOps_2d import ConvOperator

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 0
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
    "nu": 0.001, "N": 64, "L": 1.0,
    "tEnd": 0.5, "dt": 0.002,       # Nt = 250 steps
    "t_slice": 5,                    # dt_sub = 0.01, Nt_sub = 50 frames
    # IC bounds (in-dist)
    "aa_lo": 0.5,  "aa_hi": 1.0,
    "bb_lo": 0.5,  "bb_hi": 1.0,
    # IC bounds (OOD) — capped at 1.5 to avoid spectral blow-up
    "aa_lo_ood": 1.0, "aa_hi_ood": 1.5,
    "bb_lo_ood": 1.0, "bb_hi_ood": 1.5,
    # Dataset
    "n_train": 200, "n_cal": 100, "n_val": 100,
    # FNO (paper: modes=8, width=16, n_vars=4)
    "T_in": 1, "T_out": 20, "step": 1,
    "modes": 8, "width": 16, "num_vars": 4,
    # Training (paper: batch=5, epochs=500)
    "epochs": 500, "batch_size": 50,
    "lr": 5e-3, "sched_step": 100, "sched_gamma": 0.5,
    # CP / PRRS
    "alpha": 0.10, "coverage_target": 0.90,
    "n_tau_grid": 500,
}

# derived constants (module-level, used in solver and PRE)
N_      = cfg["N"]
L_      = cfg["L"]
nu_     = cfg["nu"]
dx      = L_ / N_                           # 0.015625
dt_val  = cfg["dt"]                         # 0.002
dt_sub  = cfg["dt"] * cfg["t_slice"]        # 0.01
Nt      = int(cfg["tEnd"] / cfg["dt"])      # 250

print(f"NS-2D: N={N_}, dx={dx:.5f}, dt={dt_val}, dt_sub={dt_sub}, Nt={Nt}")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PSEUDO-SPECTRAL NS SOLVER  (vorticity–stream-function)
#     ω_t + J(ψ,ω) = ν∇²ω,  ∇²ψ=-ω,  u=∂ψ/∂y,  v=-∂ψ/∂x
#     Semi-implicit CN for viscosity, explicit Euler for Jacobian.
# ═════════════════════════════════════════════════════════════════════════════

def _wavenumbers(N, L):
    k = np.fft.fftfreq(N, d=1.0/N) * (2.0 * np.pi / L)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0      # avoid /0; mean mode forced to 0 explicitly
    return KX, KY, K2


def _dealias_mask(N):
    """2/3 rule: zero Fourier modes with |k| > N/3 to prevent aliasing blow-up."""
    k_idx = np.abs(np.fft.fftfreq(N, d=1.0/N))  # integer wavenumber indices
    KX_i, KY_i = np.meshgrid(k_idx, k_idx, indexing='ij')
    return ((KX_i <= N // 3) & (KY_i <= N // 3)).astype(complex)


def solve_ns_2d(aa, bb, cfg):
    """
    Pseudo-spectral solver for 2D incompressible NS with 2/3 dealiasing.
    IC: u=aa*sin(2πy/L), v=bb*sin(2πx/L)
    Returns dict {u,v,p,w} each shape (n_frames+1, N, N).
    """
    N   = cfg["N"];  L  = cfg["L"];  nu = cfg["nu"]
    dt  = cfg["dt"]; tsl = cfg["t_slice"]
    Nt_ = int(cfg["tEnd"] / dt)
    KX, KY, K2 = _wavenumbers(N, L)
    dealias    = _dealias_mask(N)

    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    omega0 = (bb*(2*np.pi/L)*np.cos(2*np.pi*X/L)
              - aa*(2*np.pi/L)*np.cos(2*np.pi*Y/L))

    denom = 1.0 + 0.5 * nu * K2 * dt
    numer = 1.0 - 0.5 * nu * K2 * dt

    def get_uvpw(oh):
        psi = oh / K2;  psi[0,0] = 0.0
        uh  = 1j * KY * psi
        vh  = -1j * KX * psi
        u_  = np.real(np.fft.ifft2(uh))
        v_  = np.real(np.fft.ifft2(vh))
        w_  = np.real(np.fft.ifft2(oh))
        # Pressure Poisson: ∇²p = -2(ux*vy - vx*uy)
        ux = np.real(np.fft.ifft2(1j*KX*uh))
        uy = np.real(np.fft.ifft2(1j*KY*uh))
        vx = np.real(np.fft.ifft2(1j*KX*vh))
        vy = np.real(np.fft.ifft2(1j*KY*vh))
        ph = -np.fft.fft2(-2.0*(ux*vy - vx*uy)) / K2
        ph[0,0] = 0.0
        p_ = np.real(np.fft.ifft2(ph))
        return u_, v_, p_, w_

    def jacobian_hat(oh):
        # Apply dealias before computing nonlinear term (avoids aliasing blow-up)
        oh_d = oh * dealias
        psi  = oh_d / K2;  psi[0,0] = 0.0
        u_   = np.real(np.fft.ifft2(1j * KY * psi))
        v_   = np.real(np.fft.ifft2(-1j * KX * psi))
        ox   = np.real(np.fft.ifft2(1j * KX * oh_d))
        oy   = np.real(np.fft.ifft2(1j * KY * oh_d))
        return np.fft.fft2(u_*ox + v_*oy) * dealias  # dealias output too

    oh = np.fft.fft2(omega0) * dealias   # enforce dealias on IC
    u0, v0, p0, w0 = get_uvpw(oh)
    u_s, v_s, p_s, w_s = [u0], [v0], [p0], [w0]

    for n in range(Nt_):
        J  = jacobian_hat(oh)
        oh = (numer * oh - dt * J) / denom
        oh = oh * dealias               # re-enforce after each CN step
        if (n + 1) % tsl == 0:
            u_, v_, p_, w_ = get_uvpw(oh)
            u_s.append(u_); v_s.append(v_); p_s.append(p_); w_s.append(w_)

    return {"u": np.array(u_s), "v": np.array(v_s),
            "p": np.array(p_s), "w": np.array(w_s)}


# ═════════════════════════════════════════════════════════════════════════════
# 3.  DATASET GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def lhs_sample(n, lo, hi, seed=None):
    lo = np.array(lo, dtype=float); hi = np.array(hi, dtype=float)
    seed = seed if seed is not None else np.random.randint(0, 99999)
    unit = LatinHypercube(d=len(lo), seed=seed).random(n)
    return lo + unit * (hi - lo)


def generate_dataset(n, cfg, ood=False, desc="Simulating"):
    """Returns (n, 4, N, N, Nt_sub+1) tensor — vars: [u, v, p, w]."""
    lo = [cfg["aa_lo_ood"], cfg["bb_lo_ood"]] if ood else [cfg["aa_lo"],     cfg["bb_lo"]]
    hi = [cfg["aa_hi_ood"], cfg["bb_hi_ood"]] if ood else [cfg["aa_hi"],     cfg["bb_hi"]]
    params = lhs_sample(n, lo, hi)
    samples = []
    for i in tqdm(range(n), desc=desc, leave=False):
        aa, bb = params[i]
        sol = solve_ns_2d(aa, bb, cfg)
        arr = np.stack([sol["u"], sol["v"], sol["p"], sol["w"]], axis=0)  # (4,T,N,N)
        samples.append(arr)
    arr = np.array(samples, dtype=np.float32)   # (n, 4, T, N, N)
    t   = torch.tensor(arr).permute(0, 1, 3, 4, 2)  # (n, 4, N, N, T)
    return t


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MIN-MAX NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════

class MinMaxNorm:
    def __init__(self):
        self.lo = self.hi = None
    def fit(self, x):
        self.lo = x.min(); self.hi = x.max(); return self
    def encode(self, x):
        return 2.0 * (x - self.lo) / (self.hi - self.lo + 1e-8) - 1.0
    def decode(self, x):
        return (x + 1.0) / 2.0 * (self.hi - self.lo + 1e-8) + self.lo


def make_loaders(u_data, cfg, in_norm, out_norm, shuffle=True):
    T_in, T_out = cfg["T_in"], cfg["T_out"]
    a     = u_data[..., :T_in]
    u     = u_data[..., T_in: T_in+T_out]
    a_enc = in_norm.encode(a)
    u_enc = out_norm.encode(u)
    ds    = torch.utils.data.TensorDataset(a_enc, u_enc)
    ldr   = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)
    return ldr, a, u, a_enc, u_enc


# ═════════════════════════════════════════════════════════════════════════════
# 5.  FNO — 2D multi-variable  (n_vars=4)
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
        B     = x.shape[0]
        x_ft  = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, x_ft.shape[1], x_ft.shape[2],
                             x.size(-2), x.size(-1)//2+1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out_ft[:,:,:,:m1,:m2]  = self.compl_mul(x_ft[:,:,:,:m1,:m2],  self.w1)
        out_ft[:,:,:,-m1:,:m2] = self.compl_mul(x_ft[:,:,:,-m1:,:m2], self.w2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock2d(nn.Module):
    def __init__(self, modes1, modes2, n_vars, width):
        super().__init__()
        self.spec = SpectralConv2d(width, width, n_vars, modes1, modes2)
        self.mlp1 = nn.Conv3d(width, width, 1)
        self.mlp2 = nn.Conv3d(width, width, 1)
        self.w    = nn.Conv3d(width, width, 1)
        self.b    = nn.Conv3d(2, width, 1)

    def forward(self, x, grid):
        return F.gelu(self.mlp2(F.gelu(self.mlp1(self.spec(x)))) + self.w(x) + self.b(grid))


class FNO2D(nn.Module):
    def __init__(self, T_in, step, modes1, modes2, n_vars, width):
        super().__init__()
        self.T_in = T_in; self.step = step; self.n_vars = n_vars
        self.lift  = nn.Linear(T_in + 2, width)
        self.f0    = FNOBlock2d(modes1, modes2, n_vars, width)
        self.f1    = FNOBlock2d(modes1, modes2, n_vars, width)
        self.f2    = FNOBlock2d(modes1, modes2, n_vars, width)
        self.f3    = FNOBlock2d(modes1, modes2, n_vars, width)
        self.f4    = FNOBlock2d(modes1, modes2, n_vars, width)
        self.f5    = FNOBlock2d(modes1, modes2, n_vars, width)
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, step)

    def get_grid(self, shape, dev):
        bs, nv, nx, ny = shape[:4]
        gx = torch.linspace(0,1,nx).reshape(1,1,nx,1,1).expand(bs,nv,nx,ny,1)
        gy = torch.linspace(0,1,ny).reshape(1,1,1,ny,1).expand(bs,nv,nx,ny,1)
        return torch.cat([gx, gy], dim=-1).to(dev)

    def forward(self, x):
        g  = self.get_grid(x.shape, x.device)
        x  = torch.cat([x, g], dim=-1)
        x  = self.lift(x)
        x  = x.permute(0, 4, 1, 2, 3)
        g  = g.permute(0, 4, 1, 2, 3)
        x0 = self.f0(x, g)
        x  = self.f1(x0, g)
        x  = self.f2(x, g) + x0
        x1 = self.f3(x, g)
        x  = self.f4(x1, g)
        x  = self.f5(x, g) + x1
        x  = x.permute(0, 2, 3, 4, 1)
        x  = F.gelu(self.proj1(x))
        return self.proj2(x)   # (BS, n_vars, Nx, Ny, step)

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())


# ═════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def lp_loss(pred, true, p=2):
    diff = torch.norm(pred.reshape(pred.shape[0],-1) - true.reshape(true.shape[0],-1), p=p, dim=1)
    norm = torch.norm(true.reshape(true.shape[0],-1), p=p, dim=1) + 1e-8
    return (diff / norm).mean()


def train_one_epoch(model, loader, optimizer, cfg):
    model.train()
    total = 0.0
    T_out, step = cfg["T_out"], cfg["step"]
    for a_b, u_b in loader:
        a_b = a_b.to(device); u_b = u_b.to(device)
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        inp  = a_b
        for t in range(0, T_out, step):
            out  = model(inp)
            loss = loss + lp_loss(out, u_b[..., t:t+step])
            inp  = torch.cat([inp[..., step:], out], dim=-1)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict_ar(model, a_enc, out_norm, cfg):
    model.eval()
    T_out, step = cfg["T_out"], cfg["step"]
    inp = a_enc.to(device); preds = []
    for _ in range(0, T_out, step):
        out = model(inp).cpu()
        preds.append(out)
        inp = torch.cat([inp[..., step:], out.to(device)], dim=-1)
    return out_norm.decode(torch.cat(preds, dim=-1))


def relative_l2(pred, true):
    diff = (pred - true).reshape(pred.shape[0], -1)
    norm = true.reshape(true.shape[0], -1)
    return (diff.norm(2, dim=1) / (norm.norm(2, dim=1) + 1e-8)).numpy()


# ═════════════════════════════════════════════════════════════════════════════
# 7.  PRE OPERATORS — match Joint/NS_Residuals_CP.py
#     ConvOps_2d expects (BS, Nt, Nx, Ny)
#     FNO output: (N, n_vars, Nx, Ny, T_out)  → extract var, permute → (N,T,Nx,Ny)
# ═════════════════════════════════════════════════════════════════════════════

D_t     = ConvOperator('t',       1)
D_x     = ConvOperator('x',       1)
D_y     = ConvOperator('y',       1)
D_xx_yy = ConvOperator(('x','y'), 2)


def pre_score_ns(pred, normalize=False, trim=True):
    """
    Combined PRE score: continuity + momentum residuals.
    pred: (N, 4, Nx, Ny, T_out)  — decoded, var order: [u, v, p, w]
    normalize: divide by sqrt(mean(u²+v²+p²)) per sample to remove amplitude bias.
    Returns: (N,) score array
    """
    # Extract and permute to (N, T_out, Nx, Ny)
    u_f = pred[:, 0].permute(0, 3, 1, 2)
    v_f = pred[:, 1].permute(0, 3, 1, 2)
    p_f = pred[:, 2].permute(0, 3, 1, 2)

    # Continuity: ∂u/∂x + ∂v/∂y = 0
    R_cont = D_x(u_f) + D_y(v_f)

    # Momentum x: ∂u/∂t + u∂u/∂x + v∂u/∂y - ν∇²u + ∂p/∂x = 0
    R_mx = (D_t(u_f) + u_f * D_x(u_f) + v_f * D_y(u_f)
            - nu_ * D_xx_yy(u_f) + D_x(p_f))

    # Momentum y: ∂v/∂t + u∂v/∂x + v∂v/∂y - ν∇²v + ∂p/∂y = 0
    R_my = (D_t(v_f) + u_f * D_x(v_f) + v_f * D_y(v_f)
            - nu_ * D_xx_yy(v_f) + D_y(p_f))

    if trim:
        R_cont = R_cont[..., 1:-1, 1:-1, 1:-1]
        R_mx   = R_mx  [..., 1:-1, 1:-1, 1:-1]
        R_my   = R_my  [..., 1:-1, 1:-1, 1:-1]

    score = (R_cont.abs() + R_mx.abs() + R_my.abs()).mean(dim=(1, 2, 3))

    if normalize:
        # Per-sample energy: sqrt(mean(u²+v²+p²)) — removes flow amplitude bias
        u_t = u_f[..., 1:-1, 1:-1, 1:-1] if trim else u_f
        v_t = v_f[..., 1:-1, 1:-1, 1:-1] if trim else v_f
        p_t = p_f[..., 1:-1, 1:-1, 1:-1] if trim else p_f
        energy = (u_t.pow(2) + v_t.pow(2) + p_t.pow(2)).mean(dim=(1, 2, 3)).sqrt()
        score = score / (energy + 1e-8)

    return score.detach().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CP + PRRS UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def cp_quantile(scores, alpha):
    n = len(scores)
    level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    return float(np.quantile(scores, level))

def empirical_coverage(scores_test, qhat):
    return float(np.mean(scores_test <= qhat))

def prrs_calibrate(scores_cal, errors_cal, coverage_target, n_grid=500):
    taus      = np.linspace(scores_cal.min(), scores_cal.max(), n_grid)
    sel_risks = []
    coverages = []
    for tau in taus:
        mask = scores_cal <= tau
        coverages.append(mask.mean())
        sel_risks.append(errors_cal[mask].mean() if mask.sum() >= 3 else np.inf)
    sel_risks = np.array(sel_risks); coverages = np.array(coverages)
    feasible  = coverages >= coverage_target
    tau_star  = taus[np.argmax(coverages)] if not feasible.any() \
                else taus[np.argmin(np.where(feasible, sel_risks, np.inf))]
    return tau_star, {"tau_grid": taus, "coverages": coverages, "sel_risks": sel_risks}

def risk_coverage_curve(scores, errors, n_thresholds=500):
    taus = np.linspace(scores.max(), scores.min(), n_thresholds)
    covs, risks = [], []
    for tau in taus:
        mask = scores <= tau
        if mask.sum() < 2: continue
        covs.append(mask.mean()); risks.append(errors[mask].mean())
    return np.array(covs), np.array(risks)

def auc_rc(covs, risks):
    idx = np.argsort(covs)
    return float(np.trapezoid(risks[idx], covs[idx]))


# ═════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(norm_pre=False, suffix=""):
    t_total = time.time()
    norm_tag = " [PRE normalized]" if norm_pre else " [PRE raw]"
    print("=" * 68)
    print(f"PRRS Experiment — 2D Incompressible Navier-Stokes{norm_tag}")
    print("=" * 68)
    print(f"  N={cfg['N']}  nu={cfg['nu']}  dt_sub={dt_sub:.4f}  dx={dx:.5f}")
    print(f"  n_train={cfg['n_train']}  n_cal={cfg['n_cal']}  n_val={cfg['n_val']}")
    print(f"  FNO: modes={cfg['modes']} width={cfg['width']} n_vars={cfg['num_vars']}")

    # 1 — Data
    print("\n[1/8] Generating datasets …")
    t0 = time.time()
    u_train = generate_dataset(cfg["n_train"], cfg, desc="Train")
    u_cal   = generate_dataset(cfg["n_cal"],   cfg, desc="Cal")
    u_val   = generate_dataset(cfg["n_val"],   cfg, desc="Val (in-dist)")
    u_ood   = generate_dataset(cfg["n_val"],   cfg, ood=True, desc="Val (OOD)")
    print(f"  Data gen: {(time.time()-t0)/60:.1f} min")
    print(f"  train:{tuple(u_train.shape)}  cal:{tuple(u_cal.shape)}  val:{tuple(u_val.shape)}")

    # 2 — Normalise
    print("\n[2/8] Min-Max normalisation …")
    T_in, T_out = cfg["T_in"], cfg["T_out"]
    in_norm  = MinMaxNorm().fit(u_train[..., :T_in])
    out_norm = MinMaxNorm().fit(u_train[..., T_in:T_in+T_out])

    train_ldr, _,_,_,_                       = make_loaders(u_train, cfg, in_norm, out_norm)
    _, a_cal, u_cal_out, a_cal_enc, u_cal_enc = make_loaders(u_cal, cfg, in_norm, out_norm, False)
    _, a_val, u_val_out, a_val_enc, u_val_enc = make_loaders(u_val, cfg, in_norm, out_norm, False)
    _, a_ood, u_ood_out, a_ood_enc, u_ood_enc = make_loaders(u_ood, cfg, in_norm, out_norm, False)

    # 3 — Train FNO
    print(f"\n[3/8] Training FNO ({cfg['epochs']} epochs) …")
    model = FNO2D(cfg["T_in"], cfg["step"], cfg["modes"], cfg["modes"],
                  cfg["num_vars"], cfg["width"]).to(device)
    print(f"  FNO params: {model.count_params():,}")
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg["sched_step"], cfg["sched_gamma"])
    t0 = time.time(); losses = []
    for ep in tqdm(range(cfg["epochs"]), desc="Training"):
        losses.append(train_one_epoch(model, train_ldr, opt, cfg))
        sched.step()
    print(f"  Train: {(time.time()-t0)/60:.1f} min  Final loss: {losses[-1]:.4e}")
    torch.save(model.state_dict(), os.path.join(RESULTS, 'ns2d_fno.pt'))

    # 4 — Eval
    print("\n[4/8] Evaluating FNO …")
    pred_cal = predict_ar(model, a_cal_enc, out_norm, cfg)
    pred_val = predict_ar(model, a_val_enc, out_norm, cfg)
    pred_ood = predict_ar(model, a_ood_enc, out_norm, cfg)
    err_cal  = relative_l2(pred_cal, u_cal_out)
    err_val  = relative_l2(pred_val, u_val_out)
    err_ood  = relative_l2(pred_ood, u_ood_out)
    l2_val, l2_val_std = err_val.mean(), err_val.std()
    l2_ood, l2_ood_std = err_ood.mean(), err_ood.std()
    print(f"  In-dist L2: {l2_val:.3e} ± {l2_val_std:.3e}")
    print(f"  OOD     L2: {l2_ood:.3e} ± {l2_ood_std:.3e}")

    # 5 — PRE scores
    print("\n[5/8] Computing PRE scores …")
    # Always compute raw scores for diagnostics
    scores_cal_raw = pre_score_ns(pred_cal, normalize=False)
    scores_val_raw = pre_score_ns(pred_val, normalize=False)
    scores_ood_raw = pre_score_ns(pred_ood, normalize=False)
    rho_cal_raw, _ = spearmanr(scores_cal_raw, err_cal)
    rho_val_raw, _ = spearmanr(scores_val_raw, err_val)
    rho_ood_raw, _ = spearmanr(scores_ood_raw, err_ood)
    print(f"  PRE raw — cal mean:{scores_cal_raw.mean():.3e}  ρ_cal={rho_cal_raw:.4f}")
    print(f"           val mean:{scores_val_raw.mean():.3e}  ρ_val={rho_val_raw:.4f}")
    print(f"           ood mean:{scores_ood_raw.mean():.3e}  ρ_ood={rho_ood_raw:.4f}")

    if norm_pre:
        scores_cal_n = pre_score_ns(pred_cal, normalize=True)
        scores_val_n = pre_score_ns(pred_val, normalize=True)
        scores_ood_n = pre_score_ns(pred_ood, normalize=True)
        rho_cal_n, _ = spearmanr(scores_cal_n, err_cal)
        rho_val_n, _ = spearmanr(scores_val_n, err_val)
        rho_ood_n, _ = spearmanr(scores_ood_n, err_ood)
        print(f"  PRE norm— cal mean:{scores_cal_n.mean():.3e}  ρ_cal={rho_cal_n:.4f}")
        print(f"           val mean:{scores_val_n.mean():.3e}  ρ_val={rho_val_n:.4f}")
        print(f"           ood mean:{scores_ood_n.mean():.3e}  ρ_ood={rho_ood_n:.4f}")
        scores_cal = scores_cal_n
        scores_val = scores_val_n
        scores_ood = scores_ood_n
        rho_cal, rho_val, rho_ood = rho_cal_n, rho_val_n, rho_ood_n
    else:
        scores_cal = scores_cal_raw
        scores_val = scores_val_raw
        scores_ood = scores_ood_raw
        rho_cal, rho_val, rho_ood = rho_cal_raw, rho_val_raw, rho_ood_raw

    # Track 2.3 gate
    gate_pass = bool(rho_val > 0.3)
    gate_str  = ("PASS ✓ (ρ>0.3)" if gate_pass else
                 "FAIL ✗ (ρ<0.1)" if rho_val < 0.1 else "MARGINAL (0.1≤ρ≤0.3)")
    print(f"\n  [Gate 2.3] Spearman ρ ({'norm' if norm_pre else 'raw'}): "
          f"cal={rho_cal:.4f}  val={rho_val:.4f}  ood={rho_ood:.4f}")
    print(f"  → {gate_str}")

    # 6 — PRE-CP
    print("\n[6/8] PRE-CP calibration …")
    alpha    = cfg["alpha"]
    qhat     = cp_quantile(scores_cal, alpha)
    cov_val  = empirical_coverage(scores_val, qhat)
    cov_ood  = empirical_coverage(scores_ood, qhat)
    print(f"  q̂_α={qhat:.4e}  cov_val={100*cov_val:.2f}%  cov_ood={100*cov_ood:.2f}%")
    alpha_levels   = np.arange(0.05, 0.96, 0.05)
    emp_cov_val    = [empirical_coverage(scores_val, cp_quantile(scores_cal, a)) for a in alpha_levels]
    emp_cov_ood    = [empirical_coverage(scores_ood, cp_quantile(scores_cal, a)) for a in alpha_levels]

    # 7 — PRRS
    print("\n[7/8] PRRS calibration …")
    tau_star, trace = prrs_calibrate(scores_cal, err_cal, cfg["coverage_target"], cfg["n_tau_grid"])
    print(f"  τ*={tau_star:.4e}  q̂={qhat:.4e}")

    # 8 — Compare
    print("\n[8/8] Comparing methods …")
    rc_cv, rc_rv = risk_coverage_curve(scores_val, err_val)
    rc_co, rc_ro = risk_coverage_curve(scores_ood, err_ood)
    auc_val = auc_rc(rc_cv, rc_rv); auc_ood = auc_rc(rc_co, rc_ro)

    def safe_mean(e, m): return e[m].mean() if m.sum() > 0 else np.nan

    pmv = scores_val <= tau_star;  pmo = scores_ood <= tau_star
    qmv = scores_val <= qhat;      qmo = scores_ood <= qhat

    prrs_rv  = safe_mean(err_val, pmv); prrs_ro  = safe_mean(err_ood, pmo)
    precp_rv = safe_mean(err_val, qmv); precp_ro = safe_mean(err_ood, qmo)

    np.random.seed(42)
    n_acc = max(int(pmv.mean()*len(err_val)), 1)
    rand_rv = np.mean([err_val[np.random.choice(len(err_val),n_acc,False)].mean() for _ in range(50)])
    n_aco   = max(int(pmo.mean()*len(err_ood)), 1)
    rand_ro = np.mean([err_ood[np.random.choice(len(err_ood),n_aco,False)].mean() for _ in range(50)])

    print("\n" + "="*68)
    print("RESULTS — NS-2D (Incompressible Navier-Stokes)")
    print("="*68)
    print(f"\n{'Method':<28}{'Accept%':>8}{'Sel.L2':>14}{'AUC-RC':>10}")
    print("-"*62)
    for tag, mask, risk, auc in [
        ("PRE-CP (q̂)", qmv, precp_rv, auc_val),
        ("PRRS (τ*)",  pmv, prrs_rv,  "(same)"),
        ("Random",     pmv, rand_rv,  "—"),
    ]:
        r = f"{risk:.3e}" if not isinstance(risk, float) or not np.isnan(risk) else "—"
        a = f"{auc:.4f}" if isinstance(auc, float) else auc
        print(f"  [In-dist] {tag:<18}{100*mask.mean():>8.1f}%{r:>14}{a:>10}")
    print()
    for tag, mask, risk, auc in [
        ("PRE-CP (q̂)", qmo, precp_ro, auc_ood),
        ("PRRS (τ*)",  pmo, prrs_ro,  "(same)"),
        ("Random",     pmo, rand_ro,  "—"),
    ]:
        r = f"{risk:.3e}" if isinstance(risk, float) and not np.isnan(risk) else "—"
        a = f"{auc:.4f}" if isinstance(auc, float) else auc
        print(f"  [OOD]     {tag:<18}{100*mask.mean():>8.1f}%{r:>14}{a:>10}")
    print(f"\nSpearman ρ: val={rho_val:.4f}  ood={rho_ood:.4f}  gate={gate_str[:4]}")
    print(f"Total time: {(time.time()-t_total)/60:.1f} min")
    print("="*68)

    # Plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("PRRS vs PRE-CP — 2D Navier-Stokes", fontsize=13)
    ax = axes[0]
    ax.plot(rc_cv, rc_rv, 'b-', lw=2.5, label=f'RC curve (AUC={auc_val:.4f})')
    ax.scatter([qmv.mean()],[precp_rv], s=160, marker='s', c='orange', zorder=5, label='PRE-CP q̂')
    ax.scatter([pmv.mean()],[prrs_rv],  s=200, marker='*', c='red',    zorder=6, label='PRRS τ*')
    ax.scatter([pmv.mean()],[rand_rv],  s=120, marker='D', c='gray',   zorder=5, label='Random')
    ax.set_xlabel('Acceptance Rate'); ax.set_ylabel('Selective L2')
    ax.set_title('Risk-Coverage (in-dist)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(rc_co, rc_ro, 'g-', lw=2.5, label=f'RC curve (AUC={auc_ood:.4f})')
    if not np.isnan(precp_ro): ax.scatter([qmo.mean()],[precp_ro],s=160,marker='s',c='orange',zorder=5,label='PRE-CP q̂')
    if not np.isnan(prrs_ro):  ax.scatter([pmo.mean()],[prrs_ro], s=200,marker='*',c='red',   zorder=6,label='PRRS τ*')
    ax.set_xlabel('Acceptance Rate'); ax.set_ylabel('Selective L2')
    ax.set_title('Risk-Coverage (OOD)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot([0,1],[0,1],'k-',lw=2,label='Ideal')
    ax.plot(1-alpha_levels, emp_cov_val,'b--o',lw=2,ms=5,label='PRE-CP in-dist')
    ax.plot(1-alpha_levels, emp_cov_ood,'g--s',lw=2,ms=5,label='PRE-CP OOD')
    ax.set_xlabel('1−α'); ax.set_ylabel('Empirical Coverage')
    ax.set_title('Coverage Guarantee'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = axes[3]
    feasible = trace["coverages"] >= cfg["coverage_target"]
    ax.plot(trace["tau_grid"], trace["sel_risks"],'b-',lw=2,label='Sel. Risk(τ)')
    ax.axvline(tau_star,c='red',   ls='--',lw=2,label=f'PRRS τ*={tau_star:.2e}')
    ax.axvline(qhat,    c='orange',ls='--',lw=2,label=f'PRE-CP q̂={qhat:.2e}')
    ax.fill_between(trace["tau_grid"],0,trace["sel_risks"].max()*1.1,
                    where=feasible,alpha=0.1,color='green',label='Feasible')
    ax.set_xlabel('Threshold τ'); ax.set_ylabel('Selective L2')
    ax.set_title('PRRS Grid Search'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS,f'ns2d_prrs_results{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7,4))
    plt.semilogy(losses,'b-',lw=2); plt.xlabel('Epoch'); plt.ylabel('LP Loss')
    plt.title('FNO Training — NS-2D'); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS,f'ns2d_training_loss{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    score_label = 'PRE score normalized' if norm_pre else 'PRE score (cont + mom)'
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(scores_cal,bins=40,alpha=0.5,label='Cal (in-dist)', color='blue', density=True)
    ax.hist(scores_val,bins=40,alpha=0.5,label='Val (in-dist)', color='teal', density=True)
    ax.hist(scores_ood,bins=40,alpha=0.5,label='Val (OOD)',     color='red',  density=True)
    ax.axvline(qhat,    c='orange',ls='--',lw=2,label=f'PRE-CP q̂={qhat:.2e}')
    ax.axvline(tau_star,c='red',   ls='-', lw=2,label=f'PRRS τ*={tau_star:.2e}')
    ax.set_xlabel(score_label); ax.set_ylabel('Density')
    ax.set_title(f'PRE Score Distribution: NS-2D {"(normalized)" if norm_pre else "(raw)"}')
    ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS,f'ns2d_pre_scores{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved to {RESULTS}/")

    return {
        "l2_val": float(l2_val), "l2_ood": float(l2_ood),
        "coverage_val": float(cov_val), "coverage_ood": float(cov_ood),
        "tau_star": float(tau_star), "qhat": float(qhat),
        "prrs_risk_val":  float(prrs_rv)  if not np.isnan(prrs_rv)  else None,
        "precp_risk_val": float(precp_rv) if not np.isnan(precp_rv) else None,
        "prrs_risk_ood":  float(prrs_ro)  if not np.isnan(prrs_ro)  else None,
        "precp_risk_ood": float(precp_ro) if not np.isnan(precp_ro) else None,
        "rand_risk_val":  float(rand_rv),
        "rand_risk_ood":  float(rand_ro),
        "auc_rc_val": float(auc_val), "auc_rc_ood": float(auc_ood),
        "spearman_rho_cal": float(rho_cal),
        "spearman_rho_val": float(rho_val),
        "spearman_rho_ood": float(rho_ood),
        "spearman_gate_pass": gate_pass,
        # raw ρ always reported for comparison
        "spearman_rho_raw_cal": float(rho_cal_raw),
        "spearman_rho_raw_val": float(rho_val_raw),
        "spearman_rho_raw_ood": float(rho_ood_raw),
        "norm_pre": norm_pre,
    }


if __name__ == "__main__":
    import json, math
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--seeds',       type=int,   nargs='+', default=None,
                        help='Run multiple seeds (overrides --seed)')
    parser.add_argument('--norm-pre',    action='store_true',
                        help='Normalize PRE score by flow energy (fix amplitude bias)')
    parser.add_argument('--normalize-pre', action='store_true',
                        help='Alias for --norm-pre')
    parser.add_argument('--n-train',     type=int,   default=None,
                        help='Override n_train (default: cfg value)')
    parser.add_argument('--n-cal',       type=int,   default=None,
                        help='Override n_cal (default: cfg value)')
    parser.add_argument('--output-dir',  type=str,   default=None,
                        help='Directory to save results (default: PRRS/results/)')
    args = parser.parse_args()

    # Override cfg from CLI
    if args.n_train is not None:
        cfg["n_train"] = args.n_train
    if args.n_cal is not None:
        cfg["n_cal"] = args.n_cal

    # Output dir
    if args.output_dir is not None:
        RESULTS = os.path.abspath(args.output_dir)
        os.makedirs(RESULTS, exist_ok=True)

    do_norm = args.norm_pre or args.normalize_pre
    norm_tag = "_norm" if do_norm else "_raw"
    seeds = args.seeds if args.seeds is not None else [args.seed]

    all_results = []
    for seed in seeds:
        SEED = seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        suffix  = f"_seed{SEED}{norm_tag}"
        results = main(norm_pre=do_norm, suffix=suffix)
        results["seed"] = SEED

        out = os.path.join(RESULTS, f'ns2d_results{suffix}.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out}")
        all_results.append(results)

    # Multi-seed summary
    if len(seeds) > 1:
        keys = ["l2_val", "prrs_risk_val", "precp_risk_val", "rand_risk_val",
                "spearman_rho_cal", "spearman_rho_val"]
        print("\n" + "="*60)
        print(f" NS-2D SUMMARY — {len(seeds)} seeds, n_train={cfg['n_train']}")
        print("="*60)
        for k in keys:
            vals = []
            for r in all_results:
                v = r.get(k)
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        vals.append(fv)
                except (TypeError, ValueError):
                    pass
            if vals:
                m = sum(vals) / len(vals)
                s = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                print(f"  {k:30s}: {m:.4f} ± {s:.4f}")
        # Save summary
        summary = {"seeds": seeds, "n_train": cfg["n_train"], "n_cal": cfg["n_cal"],
                   "norm_pre": do_norm, "per_seed": all_results}
        sout = os.path.join(RESULTS, f'ns2d_summary{norm_tag}.json')
        with open(sout, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved → {sout}")
