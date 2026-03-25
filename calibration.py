"""
calibration.py
==============
Calibration script for the evolutionary call auction ABM.

Run from the project root:
    python calibration.py

Two calibration approaches are implemented:

  OPTION 2 — Noise Trader Proportion Calibration
  -----------------------------------------------
  Justifies the 65 ZI / 35 informed split against empirical market
  microstructure literature (Kyle, 1985; Glosten & Milgrom, 1985), which
  estimates that roughly 60-70% of order flow in equity markets originates
  from uninformed/liquidity traders rather than informed traders.

  Runs short simulations across five informed/ZI splits and measures whether
  each split produces a positive informed wealth premium — confirming that
  information remains exploitable at the chosen ratio.

  OPTION 3 — Grid Search Calibration (tent-peg via stylised facts)
  -----------------------------------------------------------------
  Varies two free parameters systematically:
    - n_zi_agents     : number of noise traders  (controls market liquidity)
    - info_param_high : upper bound of info_param (controls information asymmetry)

  For each combination, measures two stylised-fact benchmarks:
    1. Informed/ZI wealth premium > 0%
       (Gode & Sunder, 1993: informed agents must outperform ZI baseline)
    2. No-clear rate < 10%
       (Market must clear on at least 9-in-10 rounds — basic liquidity check)

  The valid region (both criteria met) is the calibrated parameter space.
  The model's baseline configuration must sit inside this region.

Outputs
-------
  calibration_results.csv  — full grid search table with pass/fail flags
  calibration_summary.txt  — human-readable summary for the report
"""

import itertools
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import run_experiment
except ImportError as e:
    print(f"\nImport error: {e}")
    print("Run this script from the project root directory.")
    sys.exit(1)


# =============================================================================
#  CONFIGURATION
# =============================================================================

CAL_GENERATIONS = 10
CAL_ROUNDS      = 25
CAL_SEED        = 42
CAL_DB_PATH     = "calibration_temp.duckdb"

# Option 2: splits to test — (n_informed, n_zi), total = 100
NOISE_PROPORTION_SPLITS = [
    (10, 90),
    (25, 75),
    (35, 65),   # BASELINE
    (50, 50),
    (65, 35),
]

# Option 3: grid axes
GRID_N_ZI    = [35, 50, 65, 80]
GRID_IP_HIGH = [0.5, 0.75, 1.0, 1.5, 2.0]

# Validity thresholds
TARGET_MIN_WEALTH_PREMIUM = 0.0
TARGET_MAX_NO_CLEAR_RATE  = 0.10


# =============================================================================
#  HELPERS
# =============================================================================

def _hr(char="-", width=70):
    print(char * width)


def _header(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}")


def _run_short(label, overrides):
    result = run_experiment(
        config_overrides={
            "experiment_name":   f"cal_{label}",
            "experiment_type":   "calibration",
            "run_notes":         "",
            "experiment_seed":   CAL_SEED,
            "n_generations":     CAL_GENERATIONS,
            "n_rounds":          CAL_ROUNDS,
            "GBM_S0":            100,
            "GBM_volatility":    0.20,
            "GBM_drift":         0.05,
            "info_param_distribution_type": "evenly_spaced",
            "signal_generator_noise_distribution": "lognormal",
            "rolling_n":         5,
            **overrides,
        },
        run_analysis=False,
        disable_db_writes=True, # <--- Bypasses all disk I/O
    )
    return result["generation_counts_df"]


def _metrics(df):
    """Extract key calibration metrics from generation_counts_df."""
    if df.empty:
        return dict(premium=float("nan"), ncr=float("nan"), vol=float("nan"))

    half = max(1, len(df) // 2)
    tail = df.tail(half)

    w_inf = tail.get("mean_wealth_parameterised_informed", pd.Series(dtype=float))
    w_zi  = tail.get("mean_wealth_zi",                    pd.Series(dtype=float))

    if w_inf.notna().any() and w_zi.notna().any():
        mi   = float(w_inf.mean())
        mz   = float(w_zi.mean())
        prem = (mi - mz) / abs(mz) * 100.0 if abs(mz) > 1e-9 else float("nan")
    else:
        prem = float("nan")

    ncr = float(tail["no_clear_rate"].mean()) if "no_clear_rate" in tail else float("nan")
    vol = float(tail["mean_volume"].mean())   if "mean_volume" in tail   else float("nan")

    return dict(premium=prem, ncr=ncr, vol=vol)


# =============================================================================
#  OPTION 2: NOISE TRADER PROPORTION CALIBRATION
# =============================================================================

def run_option2():
    _header("OPTION 2: Noise Trader Proportion Calibration")
    print("""
  Literature basis (Kyle 1985; Glosten & Milgrom 1985):
    Empirical estimates suggest approximately 60-70% of order flow in
    equity markets originates from uninformed/liquidity traders.
    The baseline split of 65 ZI / 35 informed (65% noise) sits within
    this range. We verify that the chosen ratio keeps the informed
    wealth premium positive — confirming information remains exploitable.
    """)

    rows = []
    for n_inf, n_zi in NOISE_PROPORTION_SPLITS:
        pct_noise = int(100 * n_zi / (n_zi + n_inf))
        df = _run_short(
            f"informed={n_inf}_zi={n_zi}",
            {"n_parameterised_agents": n_inf,
             "n_zi_agents":            n_zi,
             "distribution_data":      {"low": 0.0, "high": 2.0}},
        )
        m = _metrics(df)
        rows.append({
            "n_informed":         n_inf,
            "n_zi":               n_zi,
            "pct_noise":          pct_noise,
            "wealth_premium_pct": round(m["premium"], 2),
            "no_clear_rate":      round(m["ncr"], 4),
            "mean_volume":        round(m["vol"], 4),
            "premium_positive":   m["premium"] > 0,
            "BASELINE":           (n_zi == 65 and n_inf == 35),
        })

    df_out = pd.DataFrame(rows)

    _hr()
    print(f"\n  {'n_inf':>6}  {'n_zi':>5}  {'%noise':>7}  "
          f"{'premium%':>10}  {'no_clear':>9}  {'prem>0':>7}  {'BASE':>5}")
    _hr("-")
    for _, r in df_out.iterrows():
        mark = " *" if r["BASELINE"] else ""
        ok   = "YES" if r["premium_positive"] else "NO "
        print(f"  {int(r['n_informed']):>6}  {int(r['n_zi']):>5}  "
              f"{int(r['pct_noise']):>6}%  "
              f"{r['wealth_premium_pct']:>+10.2f}%  "
              f"{r['no_clear_rate']:>9.4f}  "
              f"  {ok:>7}{mark}")

    b = df_out[df_out["BASELINE"]].iloc[0]
    print(f"\n  Baseline (65 ZI / 35 informed): premium = {b['wealth_premium_pct']:+.2f}%  "
          f"{'[PASS]' if b['premium_positive'] else '[FAIL]'}")
    if df_out["premium_positive"].all():
        print("  Positive premium holds across all tested splits — baseline is robust.")
    print()
    return df_out


# =============================================================================
#  OPTION 3: GRID SEARCH CALIBRATION
# =============================================================================

def _worker_option3(args):
    """Worker function for parallel grid search."""
    n_zi, ip_high = args
    n_inf = 100 - n_zi
    
    df = _run_short(
        f"nzi{n_zi}_ip{ip_high}",
        {"n_zi_agents":            n_zi,
         "n_parameterised_agents": n_inf,
         "distribution_data":      {"low": 0.0, "high": ip_high}},
    )
    
    m = _metrics(df)
    return n_zi, ip_high, n_inf, m

def run_option3():
    _header("OPTION 3: Grid Search Calibration")
    print(f"""
  Grid:
    n_zi_agents     in {GRID_N_ZI}
    info_param_high in {GRID_IP_HIGH}

  Validity criteria (Windrum et al. 2007 indirect calibration approach):
    1. Informed/ZI wealth premium > 0%   [Gode & Sunder 1993]
    2. No-clear rate < 10%
    """)

    grid_configs = list(itertools.product(GRID_N_ZI, GRID_IP_HIGH))
    total = len(grid_configs)
    rows  = []
    
    print(f" Launching {total} calibration runs in parallel...")

    # 2. Run them in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # .map() keeps the results in the exact order we submitted them
        results = list(executor.map(_worker_option3, grid_configs))

    # 3. Process the results (now that they are all done)
    for idx, (n_zi, ip_high, n_inf, m) in enumerate(results, 1):
        baseline  = (n_zi == 65 and abs(ip_high - 2.0) < 1e-6)
        prem_ok = m["premium"] > TARGET_MIN_WEALTH_PREMIUM
        ncr_ok  = m["ncr"]     < TARGET_MAX_NO_CLEAR_RATE
        valid   = prem_ok and ncr_ok

        print(f"  [{idx:2d}/{total}]  n_zi={n_zi}  ip_high={ip_high:.2f}  "
              f"  premium={m['premium']:+.1f}%  ncr={m['ncr']:.3f}  "
              f"{'VALID' if valid else '    -'}"
              f"{'  <-- BASELINE' if baseline else ''}")

        rows.append({
            "n_zi":               n_zi,
            "n_informed":         n_inf,
            "ip_high":            ip_high,
            "wealth_premium_pct": round(m["premium"], 2),
            "no_clear_rate":      round(m["ncr"],     4),
            "mean_volume":        round(m["vol"],      4),
            "premium_ok":         prem_ok,
            "ncr_ok":             ncr_ok,
            "VALID":              valid,
            "BASELINE":           baseline,
        })

    df_out = pd.DataFrame(rows)
    valid  = df_out[df_out["VALID"]]

    _hr()
    print(f"\n  Valid configurations ({len(valid)} of {total}):\n")
    if not valid.empty:
        print(f"  {'n_zi':>5}  {'n_inf':>6}  {'ip_high':>8}  "
              f"{'premium%':>10}  {'no_clear':>9}  {'BASE':>5}")
        _hr("-")
        for _, r in valid.iterrows():
            mark = "  * BASELINE" if r["BASELINE"] else ""
            print(f"  {int(r['n_zi']):>5}  {int(r['n_informed']):>6}  "
                  f"{r['ip_high']:>8.2f}  "
                  f"{r['wealth_premium_pct']:>+10.2f}%  "
                  f"{r['no_clear_rate']:>9.4f}{mark}")

    # Pivot heatmap
    print(f"\n  Premium (%) heatmap  (* = VALID):\n")
    try:
        piv_p = df_out.pivot(index="n_zi", columns="ip_high", values="wealth_premium_pct")
        piv_v = df_out.pivot(index="n_zi", columns="ip_high", values="VALID")
        print("  n_zi \\ ip_high   " + "   ".join(f"{c:.2f}" for c in piv_p.columns))
        _hr("-")
        for nzi in piv_p.index:
            row_str = f"  {int(nzi):>14}   "
            for ip in piv_p.columns:
                v = piv_p.loc[nzi, ip]
                flag = "*" if piv_v.loc[nzi, ip] else " "
                row_str += f"  {v:>+6.1f}%{flag}"
            print(row_str)
    except Exception:
        print(df_out[["n_zi", "ip_high", "wealth_premium_pct", "VALID"]].to_string(index=False))

    b = df_out[df_out["BASELINE"]]
    if not b.empty:
        b = b.iloc[0]
        print(f"\n  Baseline (n_zi=65, ip_high=2.0): "
              f"premium={b['wealth_premium_pct']:+.2f}%  "
              f"ncr={b['no_clear_rate']:.4f}  "
              f"{'VALID' if b['VALID'] else 'INVALID'}")
        if b["VALID"]:
            print("  The baseline sits inside the valid calibrated region.")
        else:
            print("  NOTE: baseline did not satisfy criteria in this short run.")
            print("  Consider running with a larger CAL_GENERATIONS value.")

    return df_out


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  ABM Calibration  —  Evolutionary Call Auction Market Model")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("="*70)

    df2 = run_option2()
    df3 = run_option3()

    # Save outputs
    df3.to_csv("calibration_results.csv", index=False)

    with open("calibration_summary.txt", "w") as f:
        f.write("ABM Calibration Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generations per config: {CAL_GENERATIONS}  |  Rounds: {CAL_ROUNDS}\n\n")
        f.write("OPTION 2: Noise Trader Proportion\n")
        f.write("Literature: Kyle (1985), Glosten & Milgrom (1985)\n")
        f.write("Target: 60-70% noise traders; baseline = 65%\n\n")
        f.write(df2.to_string(index=False))
        f.write("\n\nOPTION 3: Grid Search\n")
        f.write("Criteria: wealth_premium > 0%, no_clear_rate < 10%\n\n")
        f.write(df3.to_string(index=False))
        valid = df3[df3["VALID"]]
        f.write(f"\n\nValid configurations ({len(valid)}):\n")
        f.write(valid[["n_zi","ip_high","wealth_premium_pct",
                        "no_clear_rate","BASELINE"]].to_string(index=False))

    print("\n  Saved: calibration_results.csv  |  calibration_summary.txt\n")


if __name__ == "__main__":
    main()