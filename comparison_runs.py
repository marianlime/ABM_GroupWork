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

from copy import deepcopy
from pathlib import Path

# analysis.py (imported via main) forces matplotlib.use("Agg") at module level.
# Import project code first, then switch to an interactive backend.
import pandas as pd
from constants import COMPARISON_PARAM_SPECS, WEALTH_INFORMED_COL, WEALTH_ZI_COL
from main import run_experiment, DEFAULT_EXPERIMENT_CONFIG  # triggers analysis.py

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")   # change to "MacOSX" or "Qt5Agg" if TkAgg unavailable


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION – edit these to control what runs
# ─────────────────────────────────────────────────────────────────────────────

# Remove any sweep name to skip it entirely.
ACTIVE_SWEEPS = [
    "population"
]

TOTAL_AGENTS  = 100
N_GENERATIONS = 100
N_ROUNDS      = 100
SMOOTH_WINDOW = 20   # rolling-mean window in generations

# Population sweep – (n_parameterised_agents, n_zi_agents) summing to TOTAL_AGENTS
POPULATION_SWEEP = [
    (20,  80),
    (50,  50),
    (80,  20),
]

# GBM drift sweep – volatility held at FIXED_VOLATILITY
DRIFT_SWEEP      = [0.00, 0.05, 0.10]
FIXED_VOLATILITY = 0.20

# GBM volatility sweep – drift held at FIXED_DRIFT
VOLATILITY_SWEEP = [0.10, 0.15, 0.20, 0.30]
FIXED_DRIFT      = 0.05

# Default population for GBM sweeps
DEFAULT_N_INFORMED = TOTAL_AGENTS // 2
DEFAULT_N_ZI       = TOTAL_AGENTS - DEFAULT_N_INFORMED

COMPARISON_OUTPUT_DIR = Path("comparison_outputs")

# COMPARISON_PARAM_SPECS, WEALTH_INFORMED_COL, WEALTH_ZI_COL imported from constants.py
# SMOOTH_WINDOW is set independently from analysis.py's _ROLL (10) — comparison plots
# use a wider window because runs are longer and we want smoother trend lines.
PARAMS = COMPARISON_PARAM_SPECS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_progress_callback(run_label: str, n_generations: int):
    def callback(event: dict):
        if event["event"] == "generation_completed":
            gen = event["generation_index"] + 1
            m   = event.get("generation_metrics", {})
            print(
                f"  [{run_label}] gen {gen:>3}/{n_generations}"
                f"  qty_agg={m.get('mean_qty_aggression', float('nan')):.3f}"
                f"  sig_agg={m.get('mean_signal_aggression', float('nan')):.3f}"
                f"  thr={m.get('mean_threshold', float('nan')):.3f}"
                f"  clip={m.get('mean_signal_clip', float('nan')):.3f}"
                f"  info={m.get('mean_info_param_parameterised_informed', float('nan')):.3f}"
            )
    return callback


def _run_single(label: str, overrides: dict) -> pd.DataFrame:
    """Run one experiment and return its generation_counts_df."""
    print(f"\n{'='*60}")
    print(f"Starting run: {label}")
    print(f"{'='*60}")
    result = run_experiment(
        config_overrides=overrides,
        progress_callback=_make_progress_callback(label, overrides.get("n_generations", N_GENERATIONS)),
        run_analysis=False,
    )
    df = result["generation_counts_df"].reset_index(drop=True)
    df["generation"] = df["generation"].astype(int)
    return df


def _smooth(series: pd.Series) -> pd.Series:
    return series.rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean()


def _plot_series(axes, runs, series_list, ylabel):
    """Shared plotting logic for both params and wealth figures."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax_idx, (col, series_label) in enumerate(series_list):
        ax = axes[ax_idx]
        for run_idx, (run_label, df) in enumerate(runs):
            color = colors[run_idx % len(colors)]
            if col not in df.columns:
                continue
            raw    = df[col].astype(float)
            smooth = _smooth(raw)
            gens   = df["generation"].values
            ax.plot(gens, raw,    color=color, alpha=0.15, linewidth=0.8)
            ax.plot(gens, smooth, color=color, linewidth=2.0, label=run_label)

        ax.set_title(series_label, fontsize=10)
        ax.set_xlabel("Generation")
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel(ylabel)
        if ax_idx == len(series_list) - 1:
            ax.legend(fontsize=8, loc="best")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_params_figure(title: str, runs: list[tuple[str, pd.DataFrame]]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 5), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    _plot_series(axes, runs, PARAMS, "Mean parameter value")
    fig.tight_layout()
    return fig


def _build_wealth_figure(title: str, runs: list[tuple[str, pd.DataFrame]]) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run_idx, (run_label, df) in enumerate(runs):
        color = colors[run_idx % len(colors)]
        if WEALTH_INFORMED_COL not in df.columns or WEALTH_ZI_COL not in df.columns:
            continue
        diff   = df[WEALTH_INFORMED_COL].astype(float) - df[WEALTH_ZI_COL].astype(float)
        smooth = _smooth(diff)
        gens   = df["generation"].values
        ax.plot(gens, diff,   color=color, alpha=0.15, linewidth=0.8)
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


def _run_and_plot_sweep(sweep_name: str, sweep_title: str, runs: list[tuple[str, pd.DataFrame]]):
    params_fig = _build_params_figure(f"{sweep_title} — parameters", runs)
    _save_figure(params_fig, f"{sweep_name}_params.png")

    wealth_fig = _build_wealth_figure(f"{sweep_title} — wealth", runs)
    _save_figure(wealth_fig, f"{sweep_name}_wealth.png")


# ─────────────────────────────────────────────────────────────────────────────
# SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

def run_population_sweep() -> list[tuple[str, pd.DataFrame]]:
    runs = []
    for n_informed, n_zi in POPULATION_SWEEP:
        label = f"informed={n_informed} zi={n_zi}"
        overrides = {
            "n_parameterised_agents": n_informed,
            "n_zi_agents":            n_zi,
            "n_generations":          N_GENERATIONS,
            "n_rounds":               N_ROUNDS,
            "GBM_drift":              FIXED_DRIFT,
            "GBM_volatility":         FIXED_VOLATILITY,
        }
        runs.append((label, _run_single(label, overrides)))
    return runs


def run_drift_sweep() -> list[tuple[str, pd.DataFrame]]:
    runs = []
    for drift in DRIFT_SWEEP:
        label = f"drift={drift:.2f}"
        overrides = {
            "n_parameterised_agents": DEFAULT_N_INFORMED,
            "n_zi_agents":            DEFAULT_N_ZI,
            "n_generations":          N_GENERATIONS,
            "n_rounds":               N_ROUNDS,
            "GBM_drift":              drift,
            "GBM_volatility":         FIXED_VOLATILITY,
        }
        runs.append((label, _run_single(label, overrides)))
    return runs


def run_volatility_sweep() -> list[tuple[str, pd.DataFrame]]:
    runs = []
    for vol in VOLATILITY_SWEEP:
        label = f"vol={vol:.2f}"
        overrides = {
            "n_parameterised_agents": DEFAULT_N_INFORMED,
            "n_zi_agents":            DEFAULT_N_ZI,
            "n_generations":          N_GENERATIONS,
            "n_rounds":               N_ROUNDS,
            "GBM_drift":              FIXED_DRIFT,
            "GBM_volatility":         vol,
        }
        runs.append((label, _run_single(label, overrides)))
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

_SWEEP_REGISTRY = {
    "population": (
        run_population_sweep,
        f"Population sweep  (drift={FIXED_DRIFT}, vol={FIXED_VOLATILITY}, total={TOTAL_AGENTS})",
    ),
    "drift": (
        run_drift_sweep,
        f"GBM drift sweep  (vol={FIXED_VOLATILITY}, informed={DEFAULT_N_INFORMED}, zi={DEFAULT_N_ZI})",
    ),
    "volatility": (
        run_volatility_sweep,
        f"GBM volatility sweep  (drift={FIXED_DRIFT}, informed={DEFAULT_N_INFORMED}, zi={DEFAULT_N_ZI})",
    ),
}


def main():
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
