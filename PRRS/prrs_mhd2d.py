#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRRS — 2D Incompressible Resistive Magnetohydrodynamics (MHD)
=============================================================
Pseudo-spectral vorticity–magnetic-potential solver with 2/3 dealiasing.
IC: Parameterized Orszag-Tang vortex (standard MHD benchmark, paper Sec. 5.3).
    ω = av*k1*(cos(k1*x)+cos(k1*y)),  A = aB*(cos(k1*y)/k1 + cos(k2*x)/k2)

PDE (incompressible 2D resistive MHD):
  ∂ω/∂t = J(ψ,ω) - J(A,j) + ν∇²ω        [vorticity + Lorentz]
  ∂A/∂t = J(ψ,A) + η∇²A                   [magnetic induction]

  J(f,g) = f_x·g_y - f_y·g_x              [Poisson bracket]
  ω = -∇²ψ   → ψ̂ = ω̂/K²               [stream function]
  j = -∇²A   → ĵ = K²·Â               [current density]
  u = ∂ψ/∂y,  v = -∂ψ/∂x               [velocity]
  Bx= ∂A/∂y, By= -∂A/∂x               [B field]

FNO output: (ω, A, u, v)  — n_vars=4
PRE equations (5):
  R1: ω_t + (u·∇)ω - (B·∇)j - ν∇²ω      [vorticity PDE]
  R2: A_t + (u·∇)A - η∇²A                [induction PDE]
  R3: ω - (∂v/∂x - ∂u/∂y)               [curl consistency]
  R4: ∂u/∂x + ∂v/∂y                      [div-free velocity]
  R5: ∂Bx/∂x + ∂By/∂y  (Bx=D_y(A),By=-D_x(A))  [div-free B]
"""

import os, sys, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
from functools import reduce
import operator

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS   = os.path.join(ROOT, 'Utils')
sys.path.insert(0, ROOT); sys.path.insert(0, UTILS)
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

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
cfg = {
    # PDE params
    "nu":  0.01, "eta": 0.01,           # kinematic viscosity, magnetic resistivity
    "N":   64,   "L":   1.0,
    "tEnd": 0.5, "dt":  0.005,          # Nt = 100 steps
    "t_slice": 5,                        # dt_sub = 0.025, Nt_sub = 20 frames
    # IC bounds (in-dist): Orszag-Tang vortex amplitude params
    "av_lo": 0.5, "av_hi": 1.5,         # vorticity amplitude scale
    "aB_lo": 0.2, "aB_hi": 0.8,         # magnetic amplitude scale
    # IC bounds (OOD): higher amplitude — kept moderate to avoid float64 blow-up
    "av_lo_ood": 1.5, "av_hi_ood": 2.0,
    "aB_lo_ood": 0.8, "aB_hi_ood": 1.2,
    # Dataset
    "n_train": 1000, "n_cal": 400, "n_val": 100,
    # FNO
    "T_in": 1, "T_out": 20, "step": 4,
    "modes": 16, "width": 32, "num_vars": 4,
    # Training
    "epochs": 500, "batch_size": 100,
    "lr": 1e-3, "sched_step": 100, "sched_gamma": 0.5,
    # CP / PRRS
    "alpha": 0.10, "coverage_target": 0.90, "n_tau_grid": 500,
    # PRE
    "n_equations": 5,
}

N_  = cfg["N"];  L_   = cfg["L"]
nu_ = cfg["nu"]; eta_ = cfg["eta"]
dx  = L_ / N_
dt_ = cfg["dt"]; dt_sub = cfg["dt"] * cfg["t_slice"]
Nt  = int(cfg["tEnd"] / cfg["dt"])
print(f"MHD-2D: N={N_}, dx={dx:.5f}, dt={dt_}, dt_sub={dt_sub}, Nt={Nt}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PSEUDO-SPECTRAL MHD SOLVER
#     ω: vorticity,  A: magnetic potential
#     CN for dissipation, explicit Jacobian with 2/3 dealiasing
# ═══════════════════════════════════════════════════════════════════════════════

def _wavenumbers(N, L):
    k = np.fft.fftfreq(N, d=1.0/N) * (2.0 * np.pi / L)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    return KX, KY, K2

def _dealias(N):
    k_idx = np.abs(np.fft.fftfreq(N, d=1.0/N))
    KX_i, KY_i = np.meshgrid(k_idx, k_idx, indexing='ij')
    return ((KX_i <= N//3) & (KY_i <= N//3)).astype(complex)


def solve_mhd_2d(av, aB, cfg, seed=None):
    """
    Pseudo-spectral 2D resistive MHD solver.
    Returns dict {w, A, u, v} each (Nt_sub+1, N, N).
    IC: Parameterized Orszag-Tang vortex (OT benchmark, paper Section 5.3).
      ω  = av * k1 * (cos(k1*x) + cos(k1*y))
      A  = aB * (cos(k1*y)/k1 + cos(k2*x)/k2)
    where k1 = 2π/L, k2 = 4π/L.  Only amplitude (av, aB) varies across samples.
    """
    N   = cfg["N"]; L = cfg["L"]
    nu  = cfg["nu"]; eta = cfg["eta"]
    dt  = cfg["dt"]; tsl = cfg["t_slice"]
    Nt_ = int(cfg["tEnd"] / dt)
    KX, KY, K2 = _wavenumbers(N, L)
    deal = _dealias(N)

    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')

    # Parameterized Orszag-Tang vortex IC
    k1 = 2.0 * np.pi / L   # fundamental wavenumber
    k2 = 2.0 * k1           # second harmonic (for By = sin(k2*x))
    # Vorticity from OT velocity field: ω = k1*av*(cos(k1*x) + cos(k1*y))
    w0 = av * k1 * (np.cos(k1 * X) + np.cos(k1 * Y))
    # Magnetic potential: A = aB*(cos(k1*y)/k1 + cos(k2*x)/k2)
    # gives Bx = ∂A/∂y = -aB*sin(k1*y), By = -∂A/∂x = aB*sin(k2*x)/2
    A0 = aB * (np.cos(k1 * Y) / k1 + np.cos(k2 * X) / k2)

    # Crank-Nicolson denominators
    denom_w = 1.0 + 0.5 * nu  * K2 * dt
    numer_w = 1.0 - 0.5 * nu  * K2 * dt
    denom_A = 1.0 + 0.5 * eta * K2 * dt
    numer_A = 1.0 - 0.5 * eta * K2 * dt

    def get_uvjw(oh, Ah):
        """Derive physical fields from spectral ω and A."""
        psi   = oh / K2;  psi[0, 0] = 0.0
        u_    = np.real(np.fft.ifft2(1j * KY * psi))
        v_    = np.real(np.fft.ifft2(-1j * KX * psi))
        w_    = np.real(np.fft.ifft2(oh))
        A_    = np.real(np.fft.ifft2(Ah))
        return u_, v_, w_, A_

    def jacobian(fh, gh):
        """Dealiased Jacobian J(f,g) = f_x·g_y - f_y·g_x in spectral space."""
        fh_d = fh * deal; gh_d = gh * deal
        fx = np.real(np.fft.ifft2(1j * KX * fh_d))
        fy = np.real(np.fft.ifft2(1j * KY * fh_d))
        gx = np.real(np.fft.ifft2(1j * KX * gh_d))
        gy = np.real(np.fft.ifft2(1j * KY * gh_d))
        result = np.fft.fft2(fx * gy - fy * gx) * deal
        return np.where(np.isfinite(result), result, 0.0)

    oh = np.fft.fft2(w0) * deal
    Ah = np.fft.fft2(A0) * deal

    u0, v0, w_init, A_init = get_uvjw(oh, Ah)
    w_s, A_s, u_s, v_s = [w_init], [A_init], [u0], [v0]

    with np.errstate(over='ignore', invalid='ignore'):
        for n in range(Nt_):
            # Clip before Jacobian to prevent OOD overflow propagation
            oh = np.where(np.isfinite(oh), oh, 0.0)
            Ah = np.where(np.isfinite(Ah), Ah, 0.0)
            # Stream function and current density
            psi_h = oh / K2;  psi_h[0, 0] = 0.0
            jh    = K2 * Ah   # j = -∇²A → ĵ = K²·Â (K²[0,0]=1, no div-by-zero issue)

            # Vorticity Jacobian: J(ψ,ω) - J(A,j)
            J_vel = jacobian(psi_h, oh)      # (u·∇)ω term (with sign convention)
            J_mag = jacobian(Ah, jh)         # Lorentz force term
            Jw    = J_vel - J_mag            # net forcing in ω equation

            # Induction Jacobian: J(ψ,A)
            J_ind = jacobian(psi_h, Ah)

            # CN update
            oh = (numer_w * oh + dt * Jw)  / denom_w
            Ah = (numer_A * Ah + dt * J_ind) / denom_A
            oh = oh * deal
            Ah = Ah * deal
            # Clip to prevent NaN propagation
            oh = np.where(np.isfinite(oh), oh, 0.0)
            Ah = np.where(np.isfinite(Ah), Ah, 0.0)

            if (n + 1) % tsl == 0:
                u_, v_, w_, A_ = get_uvjw(oh, Ah)
                w_s.append(w_); A_s.append(A_); u_s.append(u_); v_s.append(v_)

    return {"w": np.array(w_s), "A": np.array(A_s),
            "u": np.array(u_s), "v": np.array(v_s)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(n, cfg, ood=False, desc="Simulating"):
    """Returns (n, 4, N, N, Nt_sub+1) tensor — vars: [ω, A, u, v]."""
    if ood:
        av_lo, av_hi = cfg["av_lo_ood"], cfg["av_hi_ood"]
        aB_lo, aB_hi = cfg["aB_lo_ood"], cfg["aB_hi_ood"]
    else:
        av_lo, av_hi = cfg["av_lo"], cfg["av_hi"]
        aB_lo, aB_hi = cfg["aB_lo"], cfg["aB_hi"]

    rng = np.random.default_rng(SEED + (1000 if ood else 0))
    avs = rng.uniform(av_lo, av_hi, n)
    aBs = rng.uniform(aB_lo, aB_hi, n)

    samples = []
    for i in tqdm(range(n), desc=desc, leave=False):
        sol = solve_mhd_2d(avs[i], aBs[i], cfg, seed=SEED + i + (5000 if ood else 0))
        arr = np.stack([sol["w"], sol["A"], sol["u"], sol["v"]], axis=0)  # (4,T,N,N)
        samples.append(arr)
    with np.errstate(over='ignore', invalid='ignore'):
        arr = np.array(samples, dtype=np.float32)    # (n, 4, T, N, N)
    arr = np.where(np.isfinite(arr), arr, 0.0)  # handle float32 overflow from OOD
    t   = torch.tensor(arr).permute(0, 1, 3, 4, 2)  # (n, 4, N, N, T)
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

class PerVarMinMaxNorm:
    """Per-variable Min-Max normalisation — variable axis is dim 1."""
    def __init__(self): self.lo = self.hi = None
    def fit(self, x):
        # x: (n, n_vars, Nx, Ny, T)
        nv = x.shape[1]
        flat = x.permute(1, 0, 2, 3, 4).reshape(nv, -1)  # (n_vars, rest)
        self.lo = flat.min(dim=1).values  # (n_vars,)
        self.hi = flat.max(dim=1).values  # (n_vars,)
        return self
    def encode(self, x):
        lo = self.lo.reshape(1, -1, 1, 1, 1).to(x.device)
        hi = self.hi.reshape(1, -1, 1, 1, 1).to(x.device)
        return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0
    def decode(self, x):
        lo = self.lo.reshape(1, -1, 1, 1, 1).to(x.device)
        hi = self.hi.reshape(1, -1, 1, 1, 1).to(x.device)
        return (x + 1.0) / 2.0 * (hi - lo + 1e-8) + lo

def make_loaders(u_data, cfg, in_norm, out_norm, shuffle=True):
    T_in, T_out = cfg["T_in"], cfg["T_out"]
    a = u_data[..., :T_in];  u = u_data[..., T_in:T_in+T_out]
    a_enc = in_norm.encode(a); u_enc = out_norm.encode(u)
    ds  = torch.utils.data.TensorDataset(a_enc, u_enc)
    ldr = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)
    return ldr, a, u, a_enc, u_enc


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FNO — 2D multi-variable (same architecture as NS-2D, n_vars=4)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def lp_loss(pred, true, p=2):
    diff = torch.norm(pred.reshape(pred.shape[0],-1) - true.reshape(true.shape[0],-1), p=p, dim=1)
    norm = torch.norm(true.reshape(true.shape[0],-1), p=p, dim=1) + 1e-8
    return (diff / norm).mean()

def train_one_epoch(model, loader, optimizer, cfg):
    model.train(); total = 0.0
    T_out, step = cfg["T_out"], cfg["step"]
    for a_b, u_b in loader:
        a_b = a_b.to(device); u_b = u_b.to(device)
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        inp  = a_b
        T_in = cfg["T_in"]
        for t in range(0, T_out, step):
            out  = model(inp)
            loss = loss + lp_loss(out, u_b[..., t:t+step])
            inp  = torch.cat([inp[..., step:], out.detach()], dim=-1)[..., -T_in:]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def predict_ar(model, a_enc, out_norm, cfg):
    model.eval()
    T_out, step = cfg["T_out"], cfg["step"]
    inp = a_enc.to(device); preds = []
    T_in = cfg["T_in"]
    with torch.no_grad():
        for _ in range(0, T_out, step):
            out = model(inp).cpu()
            preds.append(out)
            inp = torch.cat([inp[..., step:], out.to(device)], dim=-1)[..., -T_in:]
    return out_norm.decode(torch.cat(preds, dim=-1))

def relative_l2(pred, true):
    diff = (pred - true).reshape(pred.shape[0], -1)
    norm = true.reshape(true.shape[0], -1)
    return (diff.norm(2, dim=1) / (norm.norm(2, dim=1) + 1e-8)).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PRE OPERATORS — 5 MHD residuals
#     ConvOps_2d expects (BS, Nt, Nx, Ny)
#     FNO output: (N, n_vars, Nx, Ny, T_out) — vars: [ω, A, u, v]
# ═══════════════════════════════════════════════════════════════════════════════

D_t     = ConvOperator('t',       1)
D_x     = ConvOperator('x',       1)
D_y     = ConvOperator('y',       1)
D_xx_yy = ConvOperator(('x','y'), 2)


def pre_score_mhd(pred, normalize=False, trim=True, n_eq=5):
    """
    5-equation MHD PRE score.
    pred: (N, 4, Nx, Ny, T_out)  — var order: [ω, A, u, v]
    """
    w_f = pred[:, 0].permute(0, 3, 1, 2)   # (N, T, Nx, Ny)
    A_f = pred[:, 1].permute(0, 3, 1, 2)
    u_f = pred[:, 2].permute(0, 3, 1, 2)
    v_f = pred[:, 3].permute(0, 3, 1, 2)

    # Derived B field from A (via FD on A_f)
    Bx_f = D_y(A_f)            # Bx = ∂A/∂y
    By_f = -D_x(A_f)           # By = -∂A/∂x
    # Current density j = -∇²A
    j_f  = -D_xx_yy(A_f)

    # R1: ω_t + (u·∇)ω - (B·∇)j - ν∇²ω = 0  [vorticity PDE]
    R1 = (D_t(w_f)
          + u_f * D_x(w_f) + v_f * D_y(w_f)
          - Bx_f * D_x(j_f) - By_f * D_y(j_f)
          - nu_ * D_xx_yy(w_f))

    # R2: A_t + (u·∇)A - η∇²A = 0  [induction PDE]
    R2 = (D_t(A_f)
          + u_f * D_x(A_f) + v_f * D_y(A_f)
          - eta_ * D_xx_yy(A_f))

    # R3: ω - (∂v/∂x - ∂u/∂y) = 0  [vorticity definition check]
    R3 = w_f - (D_x(v_f) - D_y(u_f))

    # R4: ∂u/∂x + ∂v/∂y = 0  [div-free velocity]
    R4 = D_x(u_f) + D_y(v_f)

    # R5: ∂Bx/∂x + ∂By/∂y = 0  [div-free B, nonzero due to FD stencil asymmetry]
    R5 = D_x(Bx_f) + D_y(By_f)

    if trim:
        sl = (slice(None), slice(1,-1), slice(1,-1), slice(1,-1))
        R1, R2, R3, R4, R5 = R1[sl], R2[sl], R3[sl], R4[sl], R5[sl]

    residuals = [R1.abs(), R2.abs()]
    if n_eq >= 3: residuals.append(R3.abs())
    if n_eq >= 4: residuals.append(R4.abs())
    if n_eq >= 5: residuals.append(R5.abs())
    score = sum(residuals).mean(dim=(1, 2, 3))

    if normalize:
        w_t = w_f[..., 1:-1, 1:-1, 1:-1] if trim else w_f
        A_t = A_f[..., 1:-1, 1:-1, 1:-1] if trim else A_f
        energy = (w_t.pow(2) + A_t.pow(2)).mean(dim=(1, 2, 3)).sqrt()
        score = score / (energy + 1e-8)

    return score.detach().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CP + PRRS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def cp_quantile(scores, alpha):
    n = len(scores)
    level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    return float(np.quantile(scores, level))

def empirical_coverage(scores_test, qhat):
    return float(np.mean(scores_test <= qhat))

def prrs_calibrate(scores_cal, errors_cal, coverage_target, n_grid=500):
    taus      = np.linspace(scores_cal.min(), scores_cal.max(), n_grid)
    sel_risks = []; coverages = []
    for tau in taus:
        mask = scores_cal <= tau
        coverages.append(mask.mean())
        sel_risks.append(errors_cal[mask].mean() if mask.sum() >= 3 else np.inf)
    sel_risks = np.array(sel_risks); coverages = np.array(coverages)
    feasible  = coverages >= coverage_target
    tau_star  = taus[np.argmax(coverages)] if not feasible.any() \
                else taus[np.argmin(np.where(feasible, sel_risks, np.inf))]
    return tau_star, {"coverages": coverages, "sel_risks": sel_risks}

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
    return float(np.trapz(risks[idx], covs[idx]))


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(norm_pre=False, suffix="", n_eq=5):
    t_total = time.time()
    norm_tag = " [PRE normalized]" if norm_pre else " [PRE raw]"
    print("=" * 68)
    print(f"PRRS Experiment — 2D Resistive MHD{norm_tag}")
    print("=" * 68)
    print(f"  N={cfg['N']}  ν={cfg['nu']}  η={cfg['eta']}  dt_sub={dt_sub:.4f}  dx={dx:.5f}")
    print(f"  n_train={cfg['n_train']}  n_cal={cfg['n_cal']}  n_val={cfg['n_val']}")
    print(f"  FNO: modes={cfg['modes']} width={cfg['width']} n_vars={cfg['num_vars']}  n_eq={n_eq}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/8] Generating datasets …")
    t0 = time.time()
    u_train = generate_dataset(cfg["n_train"], cfg, desc="Train")
    u_cal   = generate_dataset(cfg["n_cal"],   cfg, desc="Cal")
    u_val   = generate_dataset(cfg["n_val"],   cfg, desc="Val (in-dist)")
    u_ood   = generate_dataset(cfg["n_val"],   cfg, ood=True, desc="Val (OOD)")
    print(f"  Data gen: {(time.time()-t0)/60:.1f} min")
    print(f"  train:{tuple(u_train.shape)}  cal:{tuple(u_cal.shape)}  val:{tuple(u_val.shape)}")

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    print("\n[2/8] Min-Max normalisation …")
    T_in, T_out = cfg["T_in"], cfg["T_out"]
    in_norm  = PerVarMinMaxNorm().fit(u_train[..., :T_in])
    out_norm = PerVarMinMaxNorm().fit(u_train[..., T_in:T_in+T_out])
    train_ldr, _,_,_,_                       = make_loaders(u_train, cfg, in_norm, out_norm)
    _, a_cal, u_cal_out, a_cal_enc, u_cal_enc = make_loaders(u_cal, cfg, in_norm, out_norm, False)
    _, a_val, u_val_out, a_val_enc, u_val_enc = make_loaders(u_val, cfg, in_norm, out_norm, False)
    _, a_ood, u_ood_out, a_ood_enc, u_ood_enc = make_loaders(u_ood, cfg, in_norm, out_norm, False)

    # ── 3. Train FNO ──────────────────────────────────────────────────────────
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
    ckpt = os.path.join(RESULTS, f'mhd2d_fno{suffix}.pt')
    torch.save(model.state_dict(), ckpt)

    # ── 4. Eval FNO ───────────────────────────────────────────────────────────
    print("\n[4/8] Evaluating FNO …")
    pred_cal = predict_ar(model, a_cal_enc, out_norm, cfg)
    pred_val = predict_ar(model, a_val_enc, out_norm, cfg)
    pred_ood = predict_ar(model, a_ood_enc, out_norm, cfg)
    err_cal  = relative_l2(pred_cal, u_cal_out)
    err_val  = relative_l2(pred_val, u_val_out)
    err_ood  = relative_l2(pred_ood, u_ood_out)
    l2_val   = float(err_val.mean()); l2_ood = float(err_ood.mean())
    print(f"  In-dist L2: {l2_val:.3e} ± {err_val.std():.3e}")
    print(f"  OOD     L2: {l2_ood:.3e} ± {err_ood.std():.3e}")

    # Save training loss
    fig, ax = plt.subplots(figsize=(6,3))
    ax.semilogy(losses); ax.set_xlabel("Epoch"); ax.set_ylabel("LP loss")
    ax.set_title(f"MHD-2D FNO Training{suffix}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f'mhd2d_training_loss{suffix}.png'), dpi=100)
    plt.close(fig)

    # ── 5. PRE scores ─────────────────────────────────────────────────────────
    print(f"\n[5/8] Computing PRE scores (n_eq={n_eq}) …")
    scores_cal_raw = pre_score_mhd(pred_cal, normalize=False, n_eq=n_eq)
    scores_val_raw = pre_score_mhd(pred_val, normalize=False, n_eq=n_eq)
    scores_ood_raw = pre_score_mhd(pred_ood, normalize=False, n_eq=n_eq)
    rho_cal_raw, _ = spearmanr(scores_cal_raw, err_cal)
    rho_val_raw, _ = spearmanr(scores_val_raw, err_val)
    rho_ood_raw, _ = spearmanr(scores_ood_raw, err_ood)
    print(f"  PRE raw  — ρ_cal={rho_cal_raw:.4f}  ρ_val={rho_val_raw:.4f}  ρ_ood={rho_ood_raw:.4f}")

    if norm_pre:
        scores_cal = pre_score_mhd(pred_cal, normalize=True, n_eq=n_eq)
        scores_val = pre_score_mhd(pred_val, normalize=True, n_eq=n_eq)
        scores_ood = pre_score_mhd(pred_ood, normalize=True, n_eq=n_eq)
        rho_cal, _ = spearmanr(scores_cal, err_cal)
        rho_val, _ = spearmanr(scores_val, err_val)
        rho_ood, _ = spearmanr(scores_ood, err_ood)
        print(f"  PRE norm — ρ_cal={rho_cal:.4f}  ρ_val={rho_val:.4f}  ρ_ood={rho_ood:.4f}")
    else:
        scores_cal, scores_val, scores_ood = scores_cal_raw, scores_val_raw, scores_ood_raw
        rho_cal, rho_val, rho_ood = rho_cal_raw, rho_val_raw, rho_ood_raw

    gate_pass = bool(rho_cal > 0.3)
    print(f"  Gate (ρ_cal > 0.3): {'PASS ✓' if gate_pass else 'FAIL ✗'}")

    # Score plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sc, er, lab in zip(axes,
                                [scores_cal, scores_val, scores_ood],
                                [err_cal,    err_val,    err_ood],
                                ["Cal", "Val", "OOD"]):
        rho_v, _ = spearmanr(sc, er)
        ax.scatter(sc, er, s=12, alpha=0.6)
        ax.set_xlabel("PRE score"); ax.set_ylabel("Relative L2")
        ax.set_title(f"{lab}  ρ={rho_v:.3f}")
    fig.suptitle(f"MHD-2D PRE vs L2{suffix}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f'mhd2d_pre_scores{suffix}.png'), dpi=120)
    plt.close(fig)

    # ── 6. CP quantile ────────────────────────────────────────────────────────
    print("\n[6/8] Conformal prediction …")
    alpha = cfg["alpha"]
    qhat  = cp_quantile(scores_cal, alpha)
    cov_val = empirical_coverage(scores_val, qhat)
    cov_ood = empirical_coverage(scores_ood, qhat)
    print(f"  q̂={qhat:.4e}  cov_val={cov_val:.2f}  cov_ood={cov_ood:.2f}")

    # PRE-CP selective risk
    mask_precp_val = scores_val <= qhat
    precp_risk_val = float(err_val[mask_precp_val].mean()) if mask_precp_val.sum() > 0 else float('nan')
    mask_precp_ood = scores_ood <= qhat
    precp_risk_ood = float(err_ood[mask_precp_ood].mean()) if mask_precp_ood.sum() > 0 else float('nan')

    # ── 7. PRRS ───────────────────────────────────────────────────────────────
    print("\n[7/8] PRRS calibration …")
    tau_star, rc_data = prrs_calibrate(scores_cal, err_cal, cfg["coverage_target"])
    print(f"  τ*={tau_star:.4e}  q̂={qhat:.4e}  τ* vs q̂: {'higher' if tau_star > qhat else 'lower'}")

    mask_prrs_val = scores_val <= tau_star
    prrs_cov_val  = float(mask_prrs_val.mean())
    prrs_risk_val = float(err_val[mask_prrs_val].mean()) if mask_prrs_val.sum() > 0 else float('nan')
    mask_prrs_ood = scores_ood <= tau_star
    prrs_risk_ood = float(err_ood[mask_prrs_ood].mean()) if mask_prrs_ood.sum() > 0 else float('nan')

    # Random baseline (same acceptance rate)
    np.random.seed(42)
    n_accept = max(int(prrs_cov_val * len(err_val)), 1)
    rand_risk_val = float(np.mean([
        err_val[np.random.choice(len(err_val), n_accept, False)].mean()
        for _ in range(50)]))
    rand_risk_ood = float(np.mean([
        err_ood[np.random.choice(len(err_ood), n_accept, False)].mean()
        for _ in range(50)]))
    print(f"  PRRS={prrs_risk_val:.4f}  PRE-CP={precp_risk_val:.4f}  Rand={rand_risk_val:.4f}")

    # Risk-coverage curves
    covs_val, risks_val = risk_coverage_curve(scores_val, err_val)
    covs_ood, risks_ood = risk_coverage_curve(scores_ood, err_ood)
    auc_val = auc_rc(covs_val, risks_val)
    auc_ood = auc_rc(covs_ood, risks_ood)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, covs, risks, lbl in [(axes[0], covs_val, risks_val, "Val"),
                                   (axes[1], covs_ood, risks_ood, "OOD")]:
        ax.plot(covs, risks, lw=2, label="PRRS/PRE-CP")
        ax.axhline(l2_val if lbl=="Val" else l2_ood, ls='--', c='gray', label='Random')
        ax.set_xlabel("Coverage"); ax.set_ylabel("Selective Risk")
        ax.set_title(f"Risk-Coverage {lbl}  AUC={auc_rc(covs, risks):.4f}")
        ax.legend()
    fig.suptitle(f"MHD-2D PRRS Results{suffix}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f'mhd2d_prrs_results{suffix}.png'), dpi=120)
    plt.close(fig)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print("\n[8/8] Summary:")
    print(f"  FNO L2 val: {l2_val:.4f}  OOD: {l2_ood:.4f}")
    print(f"  Coverage val: {cov_val:.2f}  OOD: {cov_ood:.2f}")
    print(f"  ρ_cal={rho_cal:.4f}  ρ_ood={rho_ood:.4f}  gate={'PASS' if gate_pass else 'FAIL'}")
    print(f"  PRRS ↓ vs Rand: {(rand_risk_val - prrs_risk_val)/rand_risk_val*100:.2f}%")
    print(f"  PRRS ↓ vs CP:   {(precp_risk_val - prrs_risk_val)/precp_risk_val*100:.2f}%")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")

    return {
        "l2_val": l2_val, "l2_ood": l2_ood,
        "coverage_val": cov_val, "coverage_ood": cov_ood,
        "tau_star": float(tau_star), "qhat": float(qhat),
        "prrs_risk_val": prrs_risk_val, "precp_risk_val": precp_risk_val,
        "prrs_risk_ood": prrs_risk_ood, "precp_risk_ood": precp_risk_ood,
        "rand_risk_val": rand_risk_val, "rand_risk_ood": rand_risk_ood,
        "auc_rc_val": auc_val, "auc_rc_ood": auc_ood,
        "spearman_rho_cal": float(rho_cal), "spearman_rho_val": float(rho_val),
        "spearman_rho_ood": float(rho_ood), "spearman_gate_pass": gate_pass,
        "spearman_rho_raw_cal": float(rho_cal_raw),
        "spearman_rho_raw_val": float(rho_val_raw),
        "spearman_rho_raw_ood": float(rho_ood_raw),
        "norm_pre": norm_pre, "n_equations": n_eq,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',          type=int,  default=0)
    parser.add_argument('--seeds',         type=int,  nargs='+', default=None)
    parser.add_argument('--norm-pre',      action='store_true')
    parser.add_argument('--normalize-pre', action='store_true')
    parser.add_argument('--n-train',       type=int,  default=None)
    parser.add_argument('--n-cal',         type=int,  default=None)
    parser.add_argument('--n-equations',   type=int,  default=5)
    parser.add_argument('--output-dir',    type=str,  default=None)
    args = parser.parse_args()

    if args.n_train is not None: cfg["n_train"] = args.n_train
    if args.n_cal   is not None: cfg["n_cal"]   = args.n_cal

    if args.output_dir is not None:
        RESULTS = os.path.abspath(args.output_dir)
        os.makedirs(RESULTS, exist_ok=True)

    do_norm  = args.norm_pre or args.normalize_pre
    norm_tag = "_norm" if do_norm else "_raw"
    seeds    = args.seeds if args.seeds is not None else [args.seed]
    n_eq     = args.n_equations

    all_results = []
    for seed in seeds:
        SEED = seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        suffix  = f"_seed{SEED}{norm_tag}_neq{n_eq}"
        results = main(norm_pre=do_norm, suffix=suffix, n_eq=n_eq)
        results["seed"] = SEED

        out = os.path.join(RESULTS, f'mhd2d_results{suffix}.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out}")
        all_results.append(results)

    if len(seeds) > 1:
        keys = ["l2_val", "prrs_risk_val", "precp_risk_val", "rand_risk_val",
                "spearman_rho_cal", "spearman_rho_val", "spearman_rho_ood"]
        print("\n" + "="*60)
        print(f" MHD-2D SUMMARY — {len(seeds)} seeds  n_eq={n_eq}")
        print("="*60)
        for k in keys:
            vals = []
            for r in all_results:
                try:
                    fv = float(r.get(k, float('nan')))
                    if not math.isnan(fv): vals.append(fv)
                except (TypeError, ValueError): pass
            if vals:
                m = sum(vals)/len(vals)
                s = (sum((v-m)**2 for v in vals)/max(len(vals)-1,1))**0.5
                print(f"  {k:30s}: {m:.4f} ± {s:.4f}")
        sout = os.path.join(RESULTS, f'mhd2d_summary{norm_tag}_neq{n_eq}.json')
        with open(sout, 'w') as f:
            json.dump({"seeds": seeds, "n_eq": n_eq, "per_seed": all_results}, f, indent=2)
        print(f"\nSummary → {sout}")
