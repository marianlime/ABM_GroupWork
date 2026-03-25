"""
comparison_runs.py
==================
Run selected sweeps of the ABM experiment and plot:
  - Evolution of mean strategy parameters across generations
  - Evolution of mean wealth (informed vs ZI) across generations

Sweeps
------
  "population"  – fix total agents, vary informed/ZI split
  "drift"       – vary GBM drift, hold volatility fixed
  "volatility"  – vary GBM volatility, hold drift fixed

Select which sweeps to run by editing ACTIVE_SWEEPS below.

Usage
-----
    python comparison_runs.py

Output: PNG files saved to COMPARISON_OUTPUT_DIR, plus interactive windows.
"""

import concurrent.futures
import os
import tempfile
from pathlib import Path

import pandas as pd
from constants import WEALTH_INFORMED_COL, WEALTH_ZI_COL
from main import run_experiment  # triggers analysis.py (forces Agg backend)

import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION – edit these to control what runs
# ─────────────────────────────────────────────────────────────────────────────

# Remove any sweep name to skip it entirely.
ACTIVE_SWEEPS = [
    "population"
]

TOTAL_AGENTS  = 100
N_GENERATIONS = 100
N_ROUNDS      = 50
SMOOTH_WINDOW = 20   # rolling-mean window in generations

# Population sweep – (n_parameterised_agents, n_zi_agents) summing to TOTAL_AGENTS
POPULATION_SWEEP = [
    (20, 80),
    (30, 70),
    (40, 60),
    (50,  50),
    (60, 40),
    (70, 30),
    (80,  20),
]

# GBM drift sweep – volatility held at FIXED_VOLATILITY
DRIFT_SWEEP      = [-0.05, -0.01, 0.00, 0.01, 0.05, 0.10, 0.20]
FIXED_VOLATILITY = 0.10

# GBM volatility sweep – drift held at FIXED_DRIFT
VOLATILITY_SWEEP = [0.00, 0.01, 0.05, 0.10, 0.20, 0.50]
FIXED_DRIFT      = 0.02

# Default population for GBM sweeps
DEFAULT_N_INFORMED = TOTAL_AGENTS // 2
DEFAULT_N_ZI       = TOTAL_AGENTS - DEFAULT_N_INFORMED

COMPARISON_OUTPUT_DIR = Path("comparison_outputs")

# Colourmap for run series – sequential gradient so intensity encodes the
# swept variable (e.g. dark = low population, bright = high population).
SWEEP_COLORMAP = "viridis"

# Parameter subplot specs: (subplot_title, [(column, series_suffix_or_None), ...])
#   Single-series subplot  → series_suffix is None  → legend shows run label only
#   Multi-series subplot   → series_suffix is a short string appended to the run label
#   Linestyles cycle: solid for the first series, dashed for the second.
PARAM_SUBPLOTS = [
    ("Qty Aggression",    [("mean_qty_aggression",                    None)]),
    ("Signal Aggression", [("mean_signal_aggression",                 None)]),
    ("Info Param (informed)", [("mean_info_param_parameterised_informed", None)]),
]

# Diversity subplot specs — mirrors PARAM_SUBPLOTS but tracks std dev over generations.
STD_SUBPLOTS = [
    ("Qty Aggression σ",    [("std_qty_aggression",                    None)]),
    ("Signal Aggression σ", [("std_signal_aggression",                 None)]),
    ("Info Param σ",        [("std_info_param_parameterised_informed", None)]),
]


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER  (module-level so it is picklable by ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_worker(args: tuple) -> tuple:
    label, overrides = args
    print(f"  [{label}] starting…", flush=True)
    
    # Pass disable_db_writes=True directly to your optimized function!
    result = run_experiment(
        config_overrides=overrides,
        progress_callback=None,
        run_analysis=False,
        disable_db_writes=True 
    )
    
    df = result["generation_counts_df"].reset_index(drop=True)
    
    needed_cols = {"generation", WEALTH_INFORMED_COL, WEALTH_ZI_COL}
    for subplots in [PARAM_SUBPLOTS, STD_SUBPLOTS]:
        for _, series_specs in subplots:
            for col, _ in series_specs:
                needed_cols.add(col)
                
    keep_cols = list(needed_cols.intersection(df.columns))
    df = df[keep_cols].copy()
    
    df = df.astype(float) 
    df["generation"] = df["generation"].astype(int)
    
    print(f"  [{label}] done.", flush=True)
    return (label, df)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_parallel(run_args: list) -> list:
    """Execute a list of (label, overrides) pairs in parallel."""
    n_workers = min(len(run_args), os.cpu_count() or 1)
    print(f"Launching {len(run_args)} runs across {n_workers} worker(s)…")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_run_single_worker, run_args))
    return results


def _smooth(series: pd.Series) -> pd.Series:
    return series.rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean()


def _sweep_colors(n: int) -> list:
    """Return n evenly-spaced colours from SWEEP_COLORMAP."""
    cmap = plt.get_cmap(SWEEP_COLORMAP)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_subplot_figure(
    title: str,
    runs: list,
    subplot_specs: list,
    ylabel: str,
) -> plt.Figure:
    """Generic figure builder used by both the params and diversity figures."""
    n_subplots = len(subplot_specs)
    fig, axes = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 5), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    colors    = _sweep_colors(len(runs))
    linestyle = ["-", "--", ":", "-."]

    for ax_idx, (subplot_title, series_specs) in enumerate(subplot_specs):
        ax    = axes[ax_idx]
        multi = len(series_specs) > 1

        for run_idx, (run_label, df) in enumerate(runs):
            color = colors[run_idx]
            for spec_idx, (col, suffix) in enumerate(series_specs):
                if col not in df.columns:
                    continue
                ls           = linestyle[spec_idx % len(linestyle)]
                legend_label = f"{run_label} – {suffix}" if multi else run_label
                raw          = df[col].astype(float)
                smooth       = _smooth(raw)
                gens         = df["generation"].values
                ax.plot(gens, raw,    color=color, alpha=0.15, linewidth=0.8, linestyle=ls)
                ax.plot(gens, smooth, color=color, linewidth=2.0, linestyle=ls, label=legend_label)

        ax.set_title(subplot_title, fontsize=10)
        ax.set_xlabel("Generation")
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel(ylabel)
        if ax_idx == n_subplots - 1:
            ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig


def _build_params_figure(title: str, runs: list) -> plt.Figure:
    return _build_subplot_figure(title, runs, PARAM_SUBPLOTS, "Mean parameter value")


def _build_diversity_figure(title: str, runs: list) -> plt.Figure:
    return _build_subplot_figure(title, runs, STD_SUBPLOTS, "Parameter std dev")


def _build_wealth_figure(title: str, runs: list) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    colors = _sweep_colors(len(runs))

    for run_idx, (run_label, df) in enumerate(runs):
        color = colors[run_idx]
        if WEALTH_INFORMED_COL not in df.columns or WEALTH_ZI_COL not in df.columns:
            continue
        diff   = df[WEALTH_INFORMED_COL].astype(float) - df[WEALTH_ZI_COL].astype(float)
        smooth = _smooth(diff)
        gens   = df["generation"].values
        # Raw background intentionally omitted – per-round noise obscures trends.
        ax.plot(gens, smooth, color=color, linewidth=2.0, label=run_label)

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title("Informed wealth − ZI wealth", fontsize=10)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Wealth difference")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, filename: str) -> None:
    COMPARISON_OUTPUT_DIR.mkdir(exist_ok=True)
    path = COMPARISON_OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def _run_and_plot_sweep(sweep_name: str, sweep_title: str, runs: list):
    params_fig = _build_params_figure(f"{sweep_title} — parameters", runs)
    _save_figure(params_fig, f"{sweep_name}_params.png")

    diversity_fig = _build_diversity_figure(f"{sweep_title} — diversity", runs)
    _save_figure(diversity_fig, f"{sweep_name}_diversity.png")

    wealth_fig = _build_wealth_figure(f"{sweep_title} — wealth", runs)
    _save_figure(wealth_fig, f"{sweep_name}_wealth.png")


# ─────────────────────────────────────────────────────────────────────────────
# SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

def run_population_sweep() -> list:
    run_args = []
    for n_informed, n_zi in POPULATION_SWEEP:
        label = f"{n_informed} parametrised, {n_zi} ZI"
        overrides = {
            "n_parameterised_agents": n_informed,
            "n_zi_agents":            n_zi,
            "n_generations":          N_GENERATIONS,
            "n_rounds":               N_ROUNDS,
            "GBM_drift":              FIXED_DRIFT,
            "GBM_volatility":         FIXED_VOLATILITY,
        }
        run_args.append((label, overrides))
    return _run_parallel(run_args)


def run_drift_sweep() -> list:
    run_args = []
    for drift in DRIFT_SWEEP:
        label = f"Drift {drift:.2f}"
        overrides = {
            "n_parameterised_agents": DEFAULT_N_INFORMED,
            "n_zi_agents":            DEFAULT_N_ZI,
            "n_generations":          N_GENERATIONS,
            "n_rounds":               N_ROUNDS,
            "GBM_drift":              drift,
            "GBM_volatility":         FIXED_VOLATILITY,
        }
        run_args.append((label, overrides))
    return _run_parallel(run_args)


def run_volatility_sweep() -> list:
    run_args = []
    for vol in VOLATILITY_SWEEP:
        label = f"Vol {vol:.2f}"
        overrides = {
            "n_parameterised_agents": DEFAULT_N_INFORMED,
            "n_zi_agents":            DEFAULT_N_ZI,
            "n_generations":          N_GENERATIONS,
            "n_rounds":               N_ROUNDS,
            "GBM_drift":              FIXED_DRIFT,
            "GBM_volatility":         vol,
        }
        run_args.append((label, overrides))
    return _run_parallel(run_args)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# Defined here (not at module level) so that worker sub-processes which
# re-import this module don't attempt a display-backend switch.
_SWEEP_REGISTRY = {
    "population": (
        run_population_sweep,
        f"Population Sweep  (drift={FIXED_DRIFT}, vol={FIXED_VOLATILITY}, total={TOTAL_AGENTS})",
    ),
    "drift": (
        run_drift_sweep,
        f"GBM Drift Sweep  (vol={FIXED_VOLATILITY}, {DEFAULT_N_INFORMED} parametrised, {DEFAULT_N_ZI} ZI)",
    ),
    "volatility": (
        run_volatility_sweep,
        f"GBM Volatility Sweep  (drift={FIXED_DRIFT}, {DEFAULT_N_INFORMED} parametrised, {DEFAULT_N_ZI} ZI)",
    ),
}


def main():
    # Switch backend here, not at module level, so spawned worker processes
    # that re-import this module don't try to open a display.
    plt.switch_backend("TkAgg")   # change to "MacOSX" or "Qt5Agg" if TkAgg unavailable

    unknown = set(ACTIVE_SWEEPS) - set(_SWEEP_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown sweep(s) in ACTIVE_SWEEPS: {unknown}. "
                         f"Valid options: {list(_SWEEP_REGISTRY)}")

    for sweep_name in ACTIVE_SWEEPS:
        sweep_fn, sweep_title = _SWEEP_REGISTRY[sweep_name]
        print(f"\n{'#'*60}")
        print(f"# {sweep_name.upper()} SWEEP")
        print(f"{'#'*60}")
        runs = sweep_fn()
        _run_and_plot_sweep(sweep_name, sweep_title, runs)

    print("\nAll runs complete. Showing figures…")
    plt.show()


if __name__ == "__main__":
    main()
