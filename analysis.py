import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim.evolution import STRATEGY_ORDER


# ----------------------------
# Per-generation metric helpers
# (called from the main loop, results stored in generation_counts)
# ----------------------------

def compute_generation_mape(g) -> float:
    """Mean absolute percentage pricing error for a completed game."""
    fundamental = np.array(g.fundamental_path, dtype=float)
    n_rounds = len(fundamental) - 1
    errors = []
    for t in range(n_rounds):
        p = g.price_history.get(t)
        if p is not None and fundamental[t] > 0:
            errors.append(abs(float(p) - fundamental[t]) / fundamental[t])
    return float(np.mean(errors)) if errors else np.nan


def compute_gini(final_score) -> float:
    """Gini coefficient of terminal wealth (0 = equality, 1 = maximum inequality)."""
    wealths = np.array([float(w) for _, w in final_score], dtype=float)
    w_min = wealths.min()
    if w_min < 0:
        wealths = wealths - w_min
    total = wealths.sum()
    if total == 0:
        return np.nan
    n = len(wealths)
    wealths = np.sort(wealths)
    index = np.arange(1, n + 1)
    return float((2 * (index * wealths).sum() - (n + 1) * total) / (n * total))


def compute_no_clear_rate(g) -> float:
    """Fraction of rounds with no clearing price."""
    n_rounds = g.n_rounds
    no_clear = sum(1 for t in range(n_rounds) if g.price_history.get(t) is None)
    return no_clear / n_rounds if n_rounds > 0 else np.nan


def compute_strategy_mean_wealth(final_score, agents) -> dict:
    """Mean terminal wealth keyed by strategy type."""
    wealth_by_type = {}
    for agent_id, wealth in final_score:
        ttype = agents[agent_id].trader_type
        wealth_by_type.setdefault(ttype, []).append(float(wealth))
    return {s: float(np.mean(v)) for s, v in wealth_by_type.items()}


def compute_strategy_mean_info_param(agents) -> dict:
    """Mean info_param keyed by strategy type."""
    ip_by_type = {}
    for agent in agents.values():
        ip_by_type.setdefault(agent.trader_type, []).append(float(agent.info_param))
    return {s: float(np.mean(v)) for s, v in ip_by_type.items()}


# ----------------------------
# Main analysis function
# ----------------------------

def analyse_game_results(
    g,
    final_score,
    n_strategic_agents=None,
    title_prefix="",
    generation_counts_df=None,
):
    """
    Post-run diagnostics for the game.

    Inputs:
      g                   : game instance (must have agents, fundamental_path, price_history/order_history)
      final_score         : list of tuples (agent_id, final_wealth)
      n_strategic_agents  : optional int, if you want to label types by id cutoff
      title_prefix        : optional str for plot titles
      generation_counts_df: optional DataFrame with columns like
                            ['generation', 'zi', 'signal_following', ...,
                             'mape', 'gini', 'no_clear_rate',
                             'mean_wealth_{strategy}', 'mean_info_param_{strategy}']

    Returns:
      results_df : pandas DataFrame with per-agent outcomes + metadata
    """

    # ----- Build per-agent results table -----
    wealth_map = dict(final_score)

    rows = []
    for aid, agent in g.agents.items():
        if hasattr(agent, "trader_type"):
            ttype = agent.trader_type
        elif n_strategic_agents is not None:
            ttype = "signal_following" if aid < n_strategic_agents else "zi"
        else:
            ttype = "unknown"

        info_param = getattr(agent, "info_param", np.nan)

        rows.append({
            "agent_id": aid,
            "type": ttype,
            "info_param": float(info_param) if info_param is not None else np.nan,
            "cash_final": float(getattr(agent, "cash", np.nan)),
            "shares_final": float(getattr(agent, "shares", np.nan)),
            "wealth_final": float(wealth_map.get(aid, np.nan)),
        })

    df = pd.DataFrame(rows).sort_values("wealth_final", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # ----- Print leaderboard + summaries -----
    print("\n=== Top 10 (by final wealth) ===")
    print(df.loc[:9, ["rank", "agent_id", "type", "info_param", "wealth_final"]].to_string(index=False))

    print("\n=== Bottom 10 (by final wealth) ===")
    print(df.loc[max(len(df) - 10, 0):, ["rank", "agent_id", "type", "info_param", "wealth_final"]].to_string(index=False))

    print("\n=== Summary by type ===")
    print(df.groupby("type")["wealth_final"].agg(["count", "mean", "std", "min", "median", "max"]).to_string())

    # ----- Info parameter effect diagnostics -----
    informed = df[df["type"] != "zi"].copy()

    if len(informed) > 5 and informed["info_param"].notna().any():
        corr = informed[["info_param", "wealth_final"]].corr().iloc[0, 1]
        print(f"\n=== Non-ZI traders: corr(info_param, wealth_final) = {corr:.4f} ===")
        print("Interpretation: if info_param is noise sigma, you'd expect NEGATIVE correlation (more noise -> worse).")

        try:
            informed["info_bin"] = pd.qcut(informed["info_param"], q=5, duplicates="drop")
            bin_stats = informed.groupby("info_bin")["wealth_final"].agg(
                ["count", "mean", "std", "min", "median", "max"]
            )
            print("\n=== Non-ZI traders: wealth by info_param quintile ===")
            print(bin_stats.to_string())
        except Exception as e:
            print("\n(Binning info_param failed; likely too many identical values.)", e)
    else:
        print("\n(No sufficient non-ZI / info_param data to analyze info effects.)")

    # ----- Time-series: fundamental vs clearing price -----
    fundamental = np.array(g.fundamental_path, dtype=float)

    n_rounds = len(fundamental) - 1
    clearing = []
    for t in range(n_rounds):
        p = g.price_history.get(t, None)
        clearing.append(np.nan if p is None else float(p))
    clearing = np.array(clearing, dtype=float)

    plt.figure()
    plt.plot(range(n_rounds), fundamental[:n_rounds], label="Fundamental (S_t)")
    plt.plot(range(n_rounds), clearing, label="Clearing price")
    plt.xlabel("Round")
    plt.ylabel("Price")
    plt.title(f"{title_prefix}Fundamental vs Clearing Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    no_clear = np.isnan(clearing).sum()
    print(f"\nRounds: {n_rounds}, no-clearing rounds: {no_clear} ({no_clear / n_rounds:.1%})")

    # ----- Wealth distribution by type -----
    plt.figure()
    for ttype, sub in df.groupby("type"):
        plt.hist(sub["wealth_final"].values, bins=30, alpha=0.6, label=ttype)
    plt.xlabel("Final wealth")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}Wealth Distribution by Type")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- Wealth vs info_param by strategy type, with best-fit lines -----
    strategy_scatter_df = df[
        df["info_param"].notna() &
        df["wealth_final"].notna() &
        (df["type"] != "zi")
    ].copy()

    if not strategy_scatter_df.empty:
        plt.figure(figsize=(10, 6))

        for ttype, sub in strategy_scatter_df.groupby("type"):
            x = sub["info_param"].to_numpy(dtype=float)
            y = sub["wealth_final"].to_numpy(dtype=float)

            scatter = plt.scatter(x, y, alpha=0.7, label=ttype)

            if len(sub) >= 2 and np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, color=scatter.get_facecolor()[0])
                print(
                    f"Best-fit line for {ttype}: "
                    f"wealth_final = {slope:.4f} * info_param + {intercept:.4f}"
                )
            else:
                print(f"Not enough variation to fit line for strategy type: {ttype}")

        plt.xlabel("info_param (noise sigma)")
        plt.ylabel("Terminal wealth")
        plt.title(f"{title_prefix}Terminal Wealth vs info_param by Strategy Type")
        plt.legend(title="Strategy Type")
        plt.tight_layout()
        plt.show()
    else:
        print("\n(No sufficient non-ZI data to plot wealth vs info_param by strategy type.)")

    # ----- Evolutionary composition plot (evolvable strategies only) -----
    if generation_counts_df is not None and not generation_counts_df.empty:
        evolvable_columns = [col for col in STRATEGY_ORDER if col != "zi" and col in generation_counts_df.columns]

        if "generation" in generation_counts_df.columns and evolvable_columns:
            plot_df = generation_counts_df.set_index("generation")

            ax = plot_df[evolvable_columns].plot(
                kind="bar",
                stacked=True,
                figsize=(12, 6)
            )
            zi_count = int(generation_counts_df["zi"].iloc[0]) if "zi" in generation_counts_df.columns else 0
            ax.set_title(f"{title_prefix}Evolvable Strategy Distribution by Generation  (ZI fixed at {zi_count})")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Number of Evolvable Agents")
            ax.legend(title="Strategy Type")
            plt.tight_layout()
            plt.show()

            all_columns = (["zi"] if "zi" in generation_counts_df.columns else []) + evolvable_columns
            print("\n=== Generation strategy counts ===")
            print(generation_counts_df[["generation"] + all_columns].to_string(index=False))

    # =====================================================================
    # EVOLUTIONARY METRICS OVER GENERATIONS
    # =====================================================================

    if generation_counts_df is not None and not generation_counts_df.empty and "generation" in generation_counts_df.columns:
        gens = generation_counts_df["generation"]
        # Exclude ZI from evolutionary metrics — its count is fixed and it doesn't evolve.
        evolvable_cols = [c for c in STRATEGY_ORDER if c != "zi" and c in generation_counts_df.columns]

        # ----- Shannon entropy of evolvable strategy distribution -----
        if evolvable_cols:
            counts_matrix = generation_counts_df[evolvable_cols].values.astype(float)
            totals = counts_matrix.sum(axis=1, keepdims=True)
            totals = np.where(totals == 0, 1.0, totals)
            probs = counts_matrix / totals
            with np.errstate(divide="ignore", invalid="ignore"):
                log_probs = np.where(probs > 0, np.log(probs), 0.0)
            entropy = -(probs * log_probs).sum(axis=1)

            plt.figure()
            plt.plot(gens, entropy)
            plt.xlabel("Generation")
            plt.ylabel("Shannon Entropy (nats)")
            plt.title(f"{title_prefix}Evolvable Strategy Diversity Over Generations")
            plt.tight_layout()
            plt.show()

            max_entropy = np.log(len(evolvable_cols))
            print(f"\nFinal generation Shannon entropy (evolvable only): {entropy[-1]:.4f}  (max possible with {len(evolvable_cols)} strategies: {max_entropy:.4f})")

        # ----- MAPE over generations -----
        if "mape" in generation_counts_df.columns:
            plt.figure()
            plt.plot(gens, generation_counts_df["mape"])
            plt.xlabel("Generation")
            plt.ylabel("MAPE")
            plt.title(f"{title_prefix}Pricing Error (MAPE) Over Generations")
            plt.tight_layout()
            plt.show()

            print(f"\nFinal generation MAPE: {generation_counts_df['mape'].iloc[-1]:.4f}")

        # ----- Gini coefficient over generations -----
        if "gini" in generation_counts_df.columns:
            plt.figure()
            plt.plot(gens, generation_counts_df["gini"])
            plt.xlabel("Generation")
            plt.ylabel("Gini Coefficient")
            plt.title(f"{title_prefix}Wealth Inequality (Gini) Over Generations")
            plt.tight_layout()
            plt.show()

            print(f"\nFinal generation Gini: {generation_counts_df['gini'].iloc[-1]:.4f}")

        # ----- No-clear rate over generations -----
        if "no_clear_rate" in generation_counts_df.columns:
            plt.figure()
            plt.plot(gens, generation_counts_df["no_clear_rate"])
            plt.xlabel("Generation")
            plt.ylabel("No-Clear Rate")
            plt.title(f"{title_prefix}No-Clearing Rate Over Generations")
            plt.tight_layout()
            plt.show()

        # ----- Mean wealth per strategy over generations -----
        mean_wealth_cols = [(s, f"mean_wealth_{s}") for s in STRATEGY_ORDER if f"mean_wealth_{s}" in generation_counts_df.columns]
        if mean_wealth_cols:
            plt.figure(figsize=(12, 5))
            for strategy, col in mean_wealth_cols:
                series = generation_counts_df[col]
                if series.notna().any():
                    plt.plot(gens, series, label=strategy)
            plt.xlabel("Generation")
            plt.ylabel("Mean Terminal Wealth")
            plt.title(f"{title_prefix}Mean Wealth per Strategy Over Generations")
            plt.legend(title="Strategy Type")
            plt.tight_layout()
            plt.show()

        # ----- Mean info_param per strategy over generations (evolvable only) -----
        # ZI is excluded: it doesn't use info_param and it doesn't evolve.
        mean_ip_cols = [(s, f"mean_info_param_{s}") for s in STRATEGY_ORDER if s != "zi" and f"mean_info_param_{s}" in generation_counts_df.columns]
        if mean_ip_cols:
            plt.figure(figsize=(12, 5))
            for strategy, col in mean_ip_cols:
                series = generation_counts_df[col]
                if series.notna().any():
                    plt.plot(gens, series, label=strategy)
            plt.xlabel("Generation")
            plt.ylabel("Mean info_param")
            plt.title(f"{title_prefix}Mean info_param per Strategy Over Generations")
            plt.legend(title="Strategy Type")
            plt.tight_layout()
            plt.show()

    # =====================================================================
    # FINAL GENERATION MICROSTRUCTURE
    # =====================================================================

    if g.market_round_records:
        market_df = pd.DataFrame(g.market_round_records)

        # ----- Bid-ask spread and volume over rounds -----
        if {"best_bid", "best_ask", "round_number"}.issubset(market_df.columns):
            spread = market_df["best_ask"] - market_df["best_bid"]

            _, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(market_df["round_number"], spread)
            axes[0].set_ylabel("Bid-Ask Spread")
            axes[0].set_title(f"{title_prefix}Market Microstructure (Final Generation)")

            if "volume" in market_df.columns:
                axes[1].bar(market_df["round_number"], market_df["volume"], alpha=0.7)
                axes[1].set_ylabel("Volume")
            axes[1].set_xlabel("Round")
            plt.tight_layout()
            plt.show()

        # ----- Price return volatility ratio -----
        if "p_t" in market_df.columns:
            clearing_series = market_df.set_index("round_number")["p_t"].reindex(range(n_rounds))
            clearing_filled = clearing_series.ffill().bfill().values.astype(float)

            fund_returns = np.diff(fundamental[:n_rounds + 1]) / np.maximum(fundamental[:n_rounds], 1e-12)
            price_returns = np.diff(clearing_filled) / np.maximum(clearing_filled[:-1], 1e-12)
            valid_price_returns = price_returns[np.isfinite(price_returns)]

            fund_vol = float(np.std(fund_returns))
            price_vol = float(np.std(valid_price_returns)) if len(valid_price_returns) > 1 else np.nan
            ratio = price_vol / fund_vol if fund_vol > 0 else np.nan

            print(f"\n=== Price return volatility ratio (actual / fundamental): {ratio:.4f} ===")
            print("  > 1 → excess volatility (market overreacts to signals)")
            print("  ≈ 1 → price volatility tracks fundamentals well")
            print("  < 1 → market is dampened / under-reactive")

    # ----- Wealth trajectories by strategy type over rounds -----
    if g.agent_round_records:
        agent_df = pd.DataFrame(g.agent_round_records)
        agent_types = {aid: agent.trader_type for aid, agent in g.agents.items()}
        agent_df["trader_type"] = agent_df["agent_id"].map(agent_types)

        fundamental_array = np.array(g.fundamental_path, dtype=float)
        agent_df["wealth_t"] = (
            agent_df["cash_end"]
            + agent_df["inventory_end"] * agent_df["round_number"].map(lambda t: float(fundamental_array[t]))
        )

        wealth_traj = (
            agent_df.groupby(["round_number", "trader_type"])["wealth_t"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 5))
        for ttype, sub in wealth_traj.groupby("trader_type"):
            plt.plot(sub["round_number"], sub["wealth_t"], label=ttype)
        plt.xlabel("Round")
        plt.ylabel("Mean Wealth (mark-to-market)")
        plt.title(f"{title_prefix}Wealth Trajectories by Strategy (Final Generation)")
        plt.legend(title="Strategy Type")
        plt.tight_layout()
        plt.show()

        # ----- Fill rate per strategy -----
        active_orders = agent_df[agent_df["order_qty"] > 0]
        if not active_orders.empty:
            fill_rate = active_orders.groupby("trader_type")["fill_ratio"].mean().sort_values(ascending=False)

            print("\n=== Mean fill rate by strategy type (final generation) ===")
            print(fill_rate.to_string())

            plt.figure()
            fill_rate.plot(kind="bar")
            plt.xlabel("Strategy Type")
            plt.ylabel("Mean Fill Rate")
            plt.title(f"{title_prefix}Order Fill Rate by Strategy (Final Generation)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

    return df
