#!/usr/bin/env python3
"""
Aggregate multi-seed results for Wave 2D and NS-2D experiments.
Reads per-seed JSON files and prints mean ± std for key metrics.

Usage:
    python PRRS/aggregate_results.py
"""
import os, json, glob
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def load_jsons(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    if not files:
        return []
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def stats(vals):
    """Return (mean, std) for a list of floats, filtering out None."""
    v = [x for x in vals if x is not None]
    if not v:
        return float('nan'), float('nan')
    return float(np.mean(v)), float(np.std(v))


def report(name, records, keys):
    if not records:
        print(f"\n[{name}] — no results found")
        return
    seeds = [r.get('seed', '?') for r in records]
    print(f"\n{'='*60}")
    print(f"  {name}  (n={len(records)} seeds: {seeds})")
    print(f"{'='*60}")
    print(f"  {'Metric':<32}  {'Mean':>12}  {'Std':>10}  {'CI_lower':>10}")
    print(f"  {'-'*32}  {'-'*12}  {'-'*10}  {'-'*10}")
    for key, label, fmt in keys:
        vals = [r.get(key) for r in records]
        m, s = stats(vals)
        # 95% CI half-width = 1.96 * std/sqrt(n) — but n is small so show std
        ci_lo = m - 1.96 * s / (len(records) ** 0.5) if not np.isnan(m) else float('nan')
        print(f"  {label:<32}  {m:{fmt}}  {s:{fmt}}  {ci_lo:{fmt}}")


# ─── Wave 2D ──────────────────────────────────────────────────────────────────
wave_records = load_jsons('wave2d_results_seed*.json')
wave_keys = [
    ('l2_val',             'L2 in-dist',         '>12.5f'),
    ('l2_ood',             'L2 OOD',              '>12.5f'),
    ('coverage_val',       'Coverage val',        '>12.4f'),
    ('coverage_ood',       'Coverage OOD',        '>12.4f'),
    ('prrs_risk_val',      'PRRS risk val',       '>12.6f'),
    ('precp_risk_val',     'PRE-CP risk val',     '>12.6f'),
    ('rand_risk_val',      'Random risk val',     '>12.6f'),
    ('auc_rc_val',         'AUC-RC val',          '>12.6f'),
    ('spearman_rho_val',   'Spearman ρ val',      '>12.4f'),
    ('spearman_rho_ood',   'Spearman ρ OOD',      '>12.4f'),
]
report("Wave 2D — 5 seeds", wave_records, wave_keys)

# Compute PRRS vs random improvement
if wave_records:
    diffs = []
    for r in wave_records:
        prrs = r.get('prrs_risk_val')
        rand = r.get('rand_risk_val')
        if prrs is not None and rand is not None and rand > 0:
            diffs.append((rand - prrs) / rand * 100)
    if diffs:
        m, s = stats(diffs)
        ci_lo = m - 1.96 * s / (len(diffs) ** 0.5)
        print(f"\n  PRRS improvement vs random: {m:.3f}% ± {s:.3f}%  (CI lower: {ci_lo:.3f}%)")
        if ci_lo > 0:
            print("  → CI does NOT include 0 — improvement is statistically meaningful")
        else:
            print("  → CI INCLUDES 0 — improvement not yet statistically significant")

# ─── NS-2D normalized ─────────────────────────────────────────────────────────
ns_norm_records = load_jsons('ns2d_results_seed*_norm.json')
ns_raw_records  = load_jsons('ns2d_results_seed*_raw.json')

ns_keys = [
    ('l2_val',             'L2 in-dist',         '>12.5f'),
    ('l2_ood',             'L2 OOD',              '>12.5f'),
    ('coverage_val',       'Coverage val',        '>12.4f'),
    ('prrs_risk_val',      'PRRS risk val',       '>12.6f'),
    ('precp_risk_val',     'PRE-CP risk val',     '>12.6f'),
    ('rand_risk_val',      'Random risk val',     '>12.6f'),
    ('auc_rc_val',         'AUC-RC val',          '>12.6f'),
    ('spearman_rho_val',   'Spearman ρ val (used)', '>12.4f'),
    ('spearman_rho_ood',   'Spearman ρ OOD',       '>12.4f'),
    ('spearman_rho_raw_val','Spearman ρ raw val',   '>12.4f'),
]
report("NS-2D NORMALIZED PRE — 5 seeds", ns_norm_records, ns_keys)
report("NS-2D RAW PRE — 5 seeds",        ns_raw_records,  ns_keys)

# NS improvement
for tag, records in [("norm", ns_norm_records), ("raw", ns_raw_records)]:
    if records:
        diffs_cp = []
        diffs_rand = []
        for r in records:
            prrs = r.get('prrs_risk_val')
            cp   = r.get('precp_risk_val')
            rand = r.get('rand_risk_val')
            if prrs is not None and cp   is not None and cp   > 0:
                diffs_cp.append((cp - prrs) / cp * 100)
            if prrs is not None and rand is not None and rand > 0:
                diffs_rand.append((rand - prrs) / rand * 100)
        if diffs_cp:
            m, s = stats(diffs_cp)
            print(f"\n  NS-2D ({tag}) PRRS vs PRE-CP: {m:.3f}% ± {s:.3f}%")
        if diffs_rand:
            m, s = stats(diffs_rand)
            print(f"  NS-2D ({tag}) PRRS vs random:  {m:.3f}% ± {s:.3f}%")

# ─── Advection ────────────────────────────────────────────────────────────────
adv_records = load_jsons('advection_results_seed*.json')
adv_keys = [
    ('err_val_mean',       'L2 in-dist',          '>12.5f'),
    ('err_ood_mean',       'L2 OOD',              '>12.5f'),
    ('precp_coverage_val', 'Coverage val',        '>12.4f'),
    ('prrs_risk_val',      'PRRS risk val',       '>12.6f'),
    ('precp_risk_val',     'PRE-CP risk val',     '>12.6f'),
    ('rand_risk_val',      'Random risk val',     '>12.6f'),
    ('auc_rc_val',         'AUC-RC val',          '>12.6f'),
]
report("Advection 1D (epochs=500, n_cal=500)", adv_records, adv_keys)

print(f"\n{'='*60}")
print("  THREE KEY NUMBERS FOR PAPER STORY DECISION")
print(f"{'='*60}")

# ρ_NS_normalized
rho_ns = [r.get('spearman_rho_val') for r in ns_norm_records if r.get('spearman_rho_val') is not None]
if rho_ns:
    print(f"  ρ_NS_normalized  = {np.mean(rho_ns):.4f} ± {np.std(rho_ns):.4f}  "
          f"({'> 0 → Story B OK' if np.mean(rho_ns) > 0 else '≤ 0 → needs re-think'})")
else:
    print("  ρ_NS_normalized  = not yet available")

# Wave improvement
wave_diffs = []
for r in wave_records:
    prrs = r.get('prrs_risk_val'); rand = r.get('rand_risk_val')
    if prrs and rand and rand > 0:
        wave_diffs.append((rand - prrs) / rand * 100)
if wave_diffs:
    m, s = stats(wave_diffs)
    print(f"  Wave improvement = {m:.3f}% ± {s:.3f}%  "
          f"({'CI>0 → defendable' if m - 1.96*s/(len(wave_diffs)**0.5) > 0 else 'CI includes 0'})")
else:
    print("  Wave improvement = not yet available")

# NS improvement
ns_diffs = []
for r in ns_norm_records:
    prrs = r.get('prrs_risk_val'); rand = r.get('rand_risk_val')
    if prrs and rand and rand > 0:
        ns_diffs.append((rand - prrs) / rand * 100)
if ns_diffs:
    m, s = stats(ns_diffs)
    print(f"  NS improvement   = {m:.3f}% ± {s:.3f}%  (3.5% target)")
else:
    print("  NS improvement   = not yet available")

print()
