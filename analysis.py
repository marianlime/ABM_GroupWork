import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
                            ['generation', 'zi', 'signal_following',
                             'utility_maximiser', 'contrarian', 'adapt_sig']

    Returns:
      results_df : pandas DataFrame with per-agent outcomes + metadata
    """

    # ----- Build per-agent results table -----
    wealth_map = dict(final_score)

    rows = []
    for aid, agent in g.agents.items():
        # infer type
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
    # Analyze all non-ZI traders, since multiple strategy types may be informed.
    informed = df[df["type"] != "zi"].copy()

    if len(informed) > 5 and informed["info_param"].notna().any():
        corr = informed[["info_param", "wealth_final"]].corr().iloc[0, 1]
        print(f"\n=== Non-ZI traders: corr(info_param, wealth_final) = {corr:.4f} ===")
        print("Interpretation: if info_param is noise sigma, you'd expect NEGATIVE correlation (more noise -> worse).")

        try:
            informed["info_bin"] = pd.qcut(informed["info_param"], q=5, duplicates="drop")
            bin_stats = informed.groupby("info_bin")["wealth_final"].agg(["count", "mean", "std", "min", "median", "max"])
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

    # ----- Wealth vs info_param scatter -----
    if len(informed) > 5 and informed["info_param"].notna().any():
        plt.figure()
        plt.scatter(informed["info_param"].values, informed["wealth_final"].values)
        plt.xlabel("info_param (noise sigma)")
        plt.ylabel("Final wealth")
        plt.title(f"{title_prefix}Non-ZI: Wealth vs info_param")
        plt.tight_layout()
        plt.show()

    # ----- Evolutionary composition plot -----
    if generation_counts_df is not None and not generation_counts_df.empty:
        strategy_columns = [
            "zi",
            "signal_following",
            "utility_maximiser",
            "contrarian",
            "adapt_sig",
        ]

        available_columns = [col for col in strategy_columns if col in generation_counts_df.columns]

        if "generation" in generation_counts_df.columns and available_columns:
            plot_df = generation_counts_df.set_index("generation")

            ax = plot_df[available_columns].plot(
                kind="bar",
                stacked=True,
                figsize=(12, 6)
            )
            ax.set_title(f"{title_prefix}Strategy Distribution by Generation")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Number of Agents")
            ax.legend(title="Strategy Type")
            plt.tight_layout()
            plt.show()

            print("\n=== Generation strategy counts ===")
            print(generation_counts_df.to_string(index=False))
        else:
            print("\n(generation_counts_df provided, but required columns were missing.)")

    return df