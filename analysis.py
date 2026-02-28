import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyse_game_results(g, final_score, n_strategic_agents=None, title_prefix=""):
    """
    Post-run diagnostics for the game.

    Inputs:
      g            : game instance (must have agents, stock_path, price_history/order_history)
      final_score  : list of tuples (agent_id, final_wealth)
      n_strategic_agents: optional int, if you want to label types by id cutoff
      title_prefix : optional str for plot titles

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
    print(df.loc[len(df)-10:, ["rank", "agent_id", "type", "info_param", "wealth_final"]].to_string(index=False))

    print("\n=== Summary by type ===")
    print(df.groupby("type")["wealth_final"].agg(["count", "mean", "std", "min", "median", "max"]).to_string())

    # ----- Info parameter effect diagnostics (informed only) -----
    informed = df[df["type"] == "signal_following"].copy()
    if len(informed) > 5 and informed["info_param"].notna().any():
        # correlation (note: your info_param is "noise sigma" so higher = worse info)
        corr = informed[["info_param", "wealth_final"]].corr().iloc[0, 1]
        print(f"\n=== Informed only: corr(info_param, wealth_final) = {corr:.4f} ===")
        print("Interpretation: if info_param is noise sigma, you'd expect NEGATIVE correlation (more noise -> worse).")

        # binned means (quintiles)
        try:
            informed["info_bin"] = pd.qcut(informed["info_param"], q=5, duplicates="drop")
            bin_stats = informed.groupby("info_bin")["wealth_final"].agg(["count", "mean", "std", "min", "median", "max"])
            print("\n=== Informed only: wealth by info_param quintile ===")
            print(bin_stats.to_string())
        except Exception as e:
            print("\n(Binning info_param failed; likely too many identical values.)", e)
    else:
        print("\n(No sufficient informed/info_param data to analyze info effects.)")

    # ----- Time-series: fundamental vs clearing price -----
    fundamental = np.array(g.stock_path, dtype=float)

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
    plt.title(f"{title_prefix} Fundamental vs Clearing Price".strip())
    plt.legend()
    plt.tight_layout()
    plt.show()

    no_clear = np.isnan(clearing).sum()
    print(f"\nRounds: {n_rounds}, no-clearing rounds: {no_clear} ({no_clear/n_rounds:.1%})")

    plt.figure()
    for ttype, sub in df.groupby("type"):
        plt.hist(sub["wealth_final"].values, bins=30, alpha=0.6, label=ttype)
    plt.xlabel("Final wealth")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Wealth distribution by type".strip())
    plt.legend()
    plt.tight_layout()
    plt.show()

    if len(informed) > 5 and informed["info_param"].notna().any():
        plt.figure()
        plt.scatter(informed["info_param"].values, informed["wealth_final"].values)
        plt.xlabel("info_param (noise sigma)")
        plt.ylabel("Final wealth")
        plt.title(f"{title_prefix} Informed: wealth vs info_param".strip())
        plt.tight_layout()
        plt.show()

    return df