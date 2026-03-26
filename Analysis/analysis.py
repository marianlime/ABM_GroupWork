"""
Fast/flexible analysis of ABM run results.
"""

#--- Library Imports for Analysis and Plotting ---
from pathlib import Path 
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
from Misc.defaults import PARAM_BOUNDS
#--- Library Imports for Analysis and Plotting ---



PARAM_NAMES = list(PARAM_BOUNDS.keys())
ROLLING_WINDOW = 10   # smoothing window (generations) used throughout
_PLOT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"


# ----------------------------
# Per-generation metric helpers
# (called from the main loop, results stored in generation_counts)
# ----------------------------

def compute_strategy_mean_wealth(final_score, agents) -> dict:
    """Mean terminal wealth keyed by strategy type."""
    agent_types = {aid: agent.trader_type for aid, agent in agents.items()}
    type_list = list(set(agent_types.values()))
    type_idx = {t: i for i, t in enumerate(type_list)}
    agent_type_indices = np.array([type_idx[agent_types[aid]] for aid, _ in final_score])
    wealths = np.array([w for _, w in final_score])
    means = _compute_strategy_mean_wealth_numba(agent_type_indices, wealths, len(type_list))
    return {type_list[i]: means[i] for i in range(len(type_list))}

@njit
def _compute_strategy_mean_wealth_numba(agent_type_indices, wealths, n_types):
    sums = np.zeros(n_types)
    counts = np.zeros(n_types)
    for i in range(len(agent_type_indices)):
        idx = agent_type_indices[i]
        sums[idx] += wealths[i]
        counts[idx] += 1
    means = np.empty(n_types)
    for i in range(n_types):
        if counts[i] > 0:
            means[i] = sums[i] / counts[i]
        else:
            means[i] = np.nan
    return means

def compute_strategy_mean_info_param(agents) -> dict:
    """Mean info_param keyed by strategy type."""
    grouped_info_params = {}
    for agent in agents.values():
        grouped_info_params.setdefault(agent.trader_type, []).append(float(agent.info_param))

    return {
        trader_type: float(np.mean(values)) if values else np.nan
        for trader_type, values in grouped_info_params.items()
    }


def compute_strategy_info_param_stats(agents) -> dict:
    """
    Mean and std of info_param keyed by strategy type.

    Returns {trader_type: {"mean": float, "std": float}}.
    """
    grouped: dict[str, list[float]] = {}
    for agent in agents.values():
        grouped.setdefault(agent.trader_type, []).append(float(agent.info_param))

    return {
        trader_type: {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values)),
        } if values else {"mean": np.nan, "std": np.nan}
        for trader_type, values in grouped.items()
    }



def compute_strategy_param_stats(agents) -> dict:
    """
    Mean and std of each learnable parameter across all parameterised_informed agents.

    - Iterates over PARAM_NAMES; returns NaN stats for any parameter with no values
    - Returns {"qty_aggression": {"mean": float, "std": float}, ...}
    """
    informed_agents = [
        agent for agent in agents.values()
        if agent.trader_type == "parameterised_informed"
    ]

    param_stats = {}
    for param in PARAM_NAMES:
        values = []
        for agent in informed_agents:
            val = agent.strategy_params.get(param)
            if val is not None:
                values.append(float(val))

        if values:
            values_np = np.array(values, dtype=float)
            param_stats[param] = {
                "mean": float(np.mean(values_np)),
                "std": float(np.std(values_np)),
            }
        else:
            param_stats[param] = {
                "mean": np.nan,
                "std": np.nan,
            }

    return param_stats


# ----------------------------
# Internal helpers
# ----------------------------

def _axes_flat(fig, axes):
    """Always return a flat list of Axes regardless of subplots shape."""
    if hasattr(axes, "flatten"):
        return axes.flatten()
    return [axes]


def _colour_cycle():
    return [c["color"] for c in plt.rcParams["axes.prop_cycle"]]


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "plot"


def _save_and_close_plot(name: str) -> None:
    _PLOT_OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(_PLOT_OUTPUT_DIR / f"{_safe_slug(name)}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _format_display_df(df: pd.DataFrame, display_cols: list[str]) -> pd.DataFrame:
    display_df = df.loc[:, display_cols].copy()
    if "type" in display_df.columns:
        zi_mask = display_df["type"] == "zi"
        for col in ["qty_aggression", "signal_aggression"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(object)
                display_df.loc[zi_mask, col] = ""
    return display_df


# ----------------------------
# Main analysis function
# ----------------------------

def analyse_game_results(g, final_score, title_prefix="", generation_counts_df=None, rolling_games=None, rolling_scores=None):
    
    """
    Post-run diagnostics for the ABM run.

    - g: most-recent Game instance used for round-level plots
    - final_score: list of (agent_id, wealth) for the most-recent game
    - title_prefix: string prepended to all plot titles
    - generation_counts_df: DataFrame built by the main loop, one row per generation
    - rolling_games / rolling_scores: last N game instances and their scores; cross-sectional
      plots use N×n_agents observations and round-by-round plots are averaged across N games
    - Returns a DataFrame with per-agent outcomes and strategy_params
    """
    
    # ── Fundamentals for round-level plots ───────────────────────────────────
    fundamental = np.array(g.fundamental_path, dtype=float)

    # ── Build per-agent results table ────────────────────────────────────────
    _use_rolling = bool(rolling_games and rolling_scores)
    _n_window    = len(rolling_games) if _use_rolling else 1
    _data_label  = f"rolling {_n_window}-generation window" if (_use_rolling and _n_window > 1) else "final generation"

    if _use_rolling:
        rows = []
        for rg, rs in zip(rolling_games, rolling_scores):
            wealth_map = dict(rs)
            for aid, agent in rg.agents.items():
                ttype  = getattr(agent, "trader_type", "unknown")
                params = getattr(agent, "strategy_params", {}) or {}
                rows.append({
                    "agent_id":          aid,
                    "type":              ttype,
                    "info_param":        float(getattr(agent, "info_param", np.nan)),
                    "qty_aggression":    params.get("qty_aggression",    np.nan),
                    "signal_aggression": params.get("signal_aggression", np.nan),
                    "wealth_final":      float(wealth_map.get(aid, np.nan)),
                })
    else:
        wealth_map = dict(final_score)
        rows = []
        for aid, agent in g.agents.items():
            ttype  = getattr(agent, "trader_type", "unknown")
            params = getattr(agent, "strategy_params", {}) or {}
            rows.append({
                "agent_id":          aid,
                "type":              ttype,
                "info_param":        float(getattr(agent, "info_param", np.nan)),
                "qty_aggression":    params.get("qty_aggression",    np.nan),
                "signal_aggression": params.get("signal_aggression", np.nan),
                "cash_final":        float(getattr(agent, "cash",   np.nan)),
                "shares_final":      float(getattr(agent, "shares", np.nan)),
                "wealth_final":      float(wealth_map.get(aid, np.nan)),
            })

    df = pd.DataFrame(rows).sort_values("wealth_final", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # ── Print summaries ───────────────────────────────────────────────────────
    display_cols = ["rank", "agent_id", "type", "info_param",
                    "qty_aggression", "signal_aggression",
                    "wealth_final"]
    display_cols = [c for c in display_cols if c in df.columns]

    display_df = _format_display_df(df, display_cols)

    print(f"\n=== Top 10 (by wealth, {_data_label}) ===")
    print(display_df.loc[:9].to_string(index=False))

    print(f"\n=== Bottom 10 (by wealth, {_data_label}) ===")
    print(display_df.loc[max(len(display_df) - 10, 0):].to_string(index=False))

    print(f"\n=== Summary by type ({_data_label}) ===")
    print(df.groupby("type")["wealth_final"].agg(["count", "mean", "std", "min", "median", "max"]).to_string())

    informed_df = df[df["type"] == "parameterised_informed"].copy()
    if not informed_df.empty:
        param_cols = [p for p in PARAM_NAMES if p in informed_df.columns]
        if param_cols:
            print(f"\n=== Parameterised agent parameter summary ({_data_label}) ===")
            print(informed_df[param_cols].describe().round(4).to_string())

    non_zi = df[df["type"] != "zi"].copy()
    if len(non_zi) > 5 and non_zi["info_param"].notna().any():
        corr = non_zi[["info_param", "wealth_final"]].corr().iloc[0, 1]
        print(f"\n=== Non-ZI traders: corr(info_param, wealth_final) = {corr:.4f} ===")
        try:
            non_zi["info_bin"] = pd.qcut(non_zi["info_param"], q=5, duplicates="drop")
            print(f"\n=== Non-ZI traders: wealth by info_param quintile ({_data_label}) ===")
            print(non_zi.groupby("info_bin", observed=False)["wealth_final"].agg(
                ["count", "mean", "std", "min", "median", "max"]
            ).to_string())
        except Exception as e:
            print("\n(Binning info_param failed.)", e)

    if not informed_df.empty:
        param_cols = [p for p in PARAM_NAMES if p in informed_df.columns and informed_df[p].notna().any()]
        if param_cols:
            print(f"\n=== Parameter–wealth correlations ({_data_label}) ===")
            rows_corr = []
            for param in param_cols:
                sub = informed_df[[param, "wealth_final"]].dropna()
                if len(sub) >= 3:
                    r = sub.corr().iloc[0, 1]
                    rows_corr.append({"parameter": param, "corr_with_wealth": round(r, 4)})
            if rows_corr:
                print(pd.DataFrame(rows_corr).to_string(index=False))

    # =========================================================================
    # CROSS-SECTIONAL PLOTS
    # =========================================================================

    # ----- 1. Wealth distribution by type -----
    plt.figure()
    for ttype, sub in df.groupby("type"):
        plt.hist(sub["wealth_final"].values, bins=30, alpha=0.6, label=ttype)
    plt.xlabel("Final wealth")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}Wealth Distribution by Type ({_data_label})")
    plt.legend()
    plt.tight_layout()
    _save_and_close_plot(f"{title_prefix}wealth_distribution_{_data_label}")

    # ----- 2. Parameter distributions -----
    param_plot_df   = (informed_df[informed_df["wealth_final"].notna()].copy()
                       if not informed_df.empty else pd.DataFrame())
    available_params = ([p for p in PARAM_NAMES
                         if not param_plot_df.empty
                         and p in param_plot_df.columns
                         and param_plot_df[p].notna().any()]
                        if not param_plot_df.empty else [])

    if available_params:
        ncols = 2
        nrows = (len(available_params) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
        axf = _axes_flat(fig, axes)

        for i, param in enumerate(available_params):
            ax = axf[i]
            lo, hi, _ = PARAM_BOUNDS[param]
            vals = param_plot_df[param].dropna().to_numpy(dtype=float)
            ax.hist(vals, bins=20, range=(lo, hi), color="steelblue", alpha=0.8)
            ax.axvline(vals.mean(), color="red", linestyle="--", label=f"mean={vals.mean():.3f}")
            ax.set_xlabel(param)
            ax.set_ylabel("Count")
            ax.set_title(param)
            ax.legend(fontsize=8)

        for j in range(len(available_params), len(axf)):
            axf[j].set_visible(False)

        fig.suptitle(f"{title_prefix}Evolved Parameter Distribution ({_data_label})")
        plt.tight_layout()
        _save_and_close_plot(f"{title_prefix}parameter_distribution_{_data_label}")

    # ----- 3. Parameter & wealth correlation heatmap (all generations) -----
    if (generation_counts_df is not None and not generation_counts_df.empty):
        mean_param_cols_hm = [f"mean_{p}" for p in PARAM_NAMES
                              if f"mean_{p}" in generation_counts_df.columns]
        wealth_col_hm = "mean_wealth_parameterised_informed"
        hm_candidates = mean_param_cols_hm + (
            [wealth_col_hm] if wealth_col_hm in generation_counts_df.columns else []
        )
        # Drop constant columns (e.g. frozen params)
        hm_cols = [
            c for c in hm_candidates
            if generation_counts_df[c].notna().any()
            and generation_counts_df[c].std() > 1e-8
        ]
        if len(hm_cols) >= 2:
            corr_m = generation_counts_df[hm_cols].dropna().corr()
            labels = [c.replace("mean_", "") for c in hm_cols]
            n_hm   = len(hm_cols)
            fig, ax = plt.subplots(figsize=(max(5, n_hm), max(4, n_hm - 1)))
            im = ax.imshow(corr_m.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n_hm))
            ax.set_yticks(range(n_hm))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)
            for row in range(n_hm):
                for col in range(n_hm):
                    val = corr_m.values[row, col]
                    color = "white" if abs(val) > 0.55 else "black"
                    ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)
            n_gens = len(generation_counts_df)
            ax.set_title(f"{title_prefix}Parameter & Wealth Correlation Matrix "
                         f"(all {n_gens} generations)")
            plt.tight_layout()
            _save_and_close_plot(f"{title_prefix}parameter_wealth_correlation_matrix")

    # =========================================================================
    # EVOLUTIONARY METRICS OVER GENERATIONS
    # =========================================================================

    if (generation_counts_df is not None
            and not generation_counts_df.empty
            and "generation" in generation_counts_df.columns):

        gens   = generation_counts_df["generation"]
        colors = _colour_cycle()

        # ----- 4. Parameter means with 10-gen rolling smooth -----
        mean_param_cols = [f"mean_{p}" for p in PARAM_NAMES
                           if f"mean_{p}" in generation_counts_df.columns]
        if mean_param_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, col in enumerate(mean_param_cols):
                c      = colors[i % len(colors)]
                raw    = generation_counts_df[col]
                smooth = raw.rolling(window=ROLLING_WINDOW, min_periods=1, center=True).mean()
                label  = col.replace("mean_", "")
                ax.plot(gens, raw,    color=c, alpha=0.15, linewidth=0.8)
                ax.plot(gens, smooth, color=c, linewidth=2.0, label=label)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Mean parameter value")
            ax.set_title(f"{title_prefix}Evolved Parameter Means Over Generations "
                         f"({ROLLING_WINDOW}-gen rolling mean)")
            ax.legend(title="Parameter")
            plt.tight_layout()
            _save_and_close_plot(f"{title_prefix}parameter_means_over_generations")

        # ----- 5. Normalised parameter diversity -----
        std_param_cols = [f"std_{p}" for p in PARAM_NAMES
                          if f"std_{p}" in generation_counts_df.columns]
        if std_param_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, col in enumerate(std_param_cols):
                c           = colors[i % len(colors)]
                param       = col.replace("std_", "")
                lo, hi, _   = PARAM_BOUNDS[param]
                raw_norm    = generation_counts_df[col] / (hi - lo)
                smooth_norm = raw_norm.rolling(window=ROLLING_WINDOW, min_periods=1, center=True).mean()
                ax.plot(gens, raw_norm,    color=c, alpha=0.15, linewidth=0.8)
                ax.plot(gens, smooth_norm, color=c, linewidth=2.0,
                        linestyle="--", label=param)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Std / parameter range  (0–1)")
            ax.set_title(f"{title_prefix}Normalised Parameter Diversity Over Generations")
            ax.legend(title="Parameter")
            plt.tight_layout()
            _save_and_close_plot(f"{title_prefix}parameter_diversity_over_generations")

        # ----- 6. Informed/ZI wealth premium -----
        w_inf = generation_counts_df.get("mean_wealth_parameterised_informed")
        w_zi  = generation_counts_df.get("mean_wealth_zi")

        if w_inf is not None and w_zi is not None:
            raw_pct  = ((w_inf - w_zi) / w_zi.abs().replace(0, np.nan)) * 100.0
            smoothed = raw_pct.rolling(window=ROLLING_WINDOW, min_periods=1, center=True).mean()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(gens, raw_pct,  color="steelblue", alpha=0.25, linewidth=0.8)
            ax.plot(gens, smoothed, color="steelblue", linewidth=2.0,
                    label=f"{ROLLING_WINDOW}-gen rolling mean")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Wealth premium (%)")
            ax.set_title(f"{title_prefix}Informed Agents' Wealth Premium Over ZI (%)")
            ax.legend()
            plt.tight_layout()
            _save_and_close_plot(f"{title_prefix}wealth_premium_over_generations")

            print(f"\nFinal-generation informed/ZI wealth premium: {raw_pct.iloc[-1]:.1f}%  "
                  f"({ROLLING_WINDOW}-gen smoothed: {smoothed.iloc[-1]:.1f}%)")

        # ----- 7. Mean info_param over generations -----
        mean_ip_cols = [c for c in generation_counts_df.columns
                        if c.startswith("mean_info_param_") and "zi" not in c]
        if mean_ip_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in mean_ip_cols:
                label  = col.replace("mean_info_param_", "")
                series = generation_counts_df[col]
                if series.notna().any():
                    ax.plot(gens, series, label=label)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Mean info_param")
            ax.set_title(f"{title_prefix}Mean info_param Over Generations")
            ax.legend(title="Strategy Type")
            plt.tight_layout()
            _save_and_close_plot(f"{title_prefix}mean_info_param_over_generations")

    # =========================================================================
    # ROUND-BY-ROUND PLOTS  (averaged over rolling window)
    # =========================================================================

    # Build aggregated agent DataFrame
    if _use_rolling:
        _all_agent = []
        for rg in rolling_games:
            _fund  = np.array(rg.fundamental_path, dtype=float)
            _types = {aid: agent.trader_type for aid, agent in rg.agents.items()}
            for rec in rg.agent_round_records:
                t = rec["round_number"]
                _all_agent.append({
                    **rec,
                    "trader_type": _types.get(rec["agent_id"], "unknown"),
                    "wealth_t":    rec["cash_end"] + rec["inventory_end"] * float(_fund[t]),
                })
        agent_df = pd.DataFrame(_all_agent)
    else:
        if g.agent_round_records:
            agent_df = pd.DataFrame(g.agent_round_records)
            agent_df["trader_type"] = agent_df["agent_id"].map(
                {aid: agent.trader_type for aid, agent in g.agents.items()})
            _fund = np.array(g.fundamental_path, dtype=float)
            agent_df["wealth_t"] = (
                agent_df["cash_end"]
                + agent_df["inventory_end"] * _fund[agent_df["round_number"].to_numpy(int)]
            )
        else:
            agent_df = pd.DataFrame()

    # ----- 8. Wealth trajectories with ±1 std bands -----
    if not agent_df.empty and "wealth_t" in agent_df.columns:
        wealth_stats = (
            agent_df.groupby(["round_number", "trader_type"])["wealth_t"]
            .agg(["mean", "std"])
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for ttype, sub in wealth_stats.groupby("trader_type"):
            sub  = sub.sort_values("round_number")
            rds  = sub["round_number"].values
            mn   = sub["mean"].values
            sd   = sub["std"].fillna(0).values
            line, = ax.plot(rds, mn, linewidth=2.0, label=ttype)
            ax.fill_between(rds, mn - sd, mn + sd, alpha=0.15, color=line.get_color())
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Wealth ± 1 std (mark-to-market)")
        ax.set_title(f"{title_prefix}Wealth Trajectories by Strategy ({_data_label})")
        ax.legend(title="Strategy Type")
        plt.tight_layout()
        _save_and_close_plot(f"{title_prefix}wealth_trajectories_{_data_label}")

    # ----- 9. Mean volume over generations -----
    if (generation_counts_df is not None and not generation_counts_df.empty
            and "mean_volume" in generation_counts_df.columns
            and "generation" in generation_counts_df.columns):
        gens = generation_counts_df["generation"]
        smooth_vol = generation_counts_df["mean_volume"].rolling(
            window=ROLLING_WINDOW, min_periods=1, center=True).mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(gens, generation_counts_df["mean_volume"], color="teal", alpha=0.2, linewidth=0.8)
        ax.plot(gens, smooth_vol, color="teal", linewidth=2.0, label=f"{ROLLING_WINDOW}-gen rolling mean")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean volume per round")
        ax.set_title(f"{title_prefix}Mean Volume Over Generations ({ROLLING_WINDOW}-gen smooth)")
        ax.legend()
        plt.tight_layout()
        _save_and_close_plot(f"{title_prefix}mean_volume_over_generations")

    return df
