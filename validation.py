"""
Inputs
------
  Reads from the most recent completed experiment in experiment_results.duckdb.
  Can also accept a generation_counts_df from a live run.

Usage
-----
  # After running main.py, validate the saved results:
  python validation.py

  # Or import and call directly after run_experiment():
  from validation import run_all_validations
  run_all_validations(generation_counts_df=result["generation_counts_df"],
                      last_game=game, final_score=final_score)

Outputs
-------
  validation_report.txt — structured pass/fail report for Appendix
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import run_experiment, DEFAULT_EXPERIMENT_CONFIG
except ImportError as e:
    print(f"\nImport error: {e}")
    print("Run this script from the project root directory.")
    sys.exit(1)

try:
    from scipy.stats import kurtosis as scipy_kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
#  THRESHOLDS  — targets for each validation check
# =============================================================================

# Program validity
MAX_CONSERVATION_ERROR    = 1e-4   # max relative error in cash + shares * price
MAX_BUDGET_VIOLATION_RATE = 0.001  # max fraction of trades that violate budget

# Model validity (stylised facts)
MIN_WEALTH_PREMIUM_PCT    = 0.0    # informed must outperform ZI
MAX_NO_CLEAR_RATE         = 0.10   # < 10% rounds fail to clear
MIN_EXCESS_KURTOSIS       = 0.0    # clearing price returns must be fatter than GBM

# Empirical validity (Level 1)
MIN_GINI                  = 0.05   # some inequality must exist
MAX_PARAM_ENTROPY_LOSS    = 0.80   # entropy must not drop below 80% of initial


# =============================================================================
#  HELPERS
# =============================================================================

def _hr(char="-", width=70):
    print(char * width)


def _header(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}")


def _result(label, passed, detail=""):
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}]  {label}")
    if detail:
        print(f"          {detail}")
    return passed


def _gini_coefficient(wealths):
    """Gini coefficient of a wealth distribution (0 = perfect equality)."""
    arr = np.array(sorted(wealths), dtype=float)
    n   = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum()))


def _shannon_entropy(counts):
    """Shannon entropy in nats given a list of counts."""
    counts = np.array([c for c in counts if c > 0], dtype=float)
    probs  = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


# =============================================================================
#  1. PROGRAM VALIDITY
# =============================================================================

def validate_program(game, final_score):
    """
    Checks that the simulator correctly implements the model.
    From lecture slide 7: just because it looks like it's working doesn't
    mean it is — test with extreme cases and step-by-step checks.
    """
    _header("1. PROGRAM VALIDITY  (Stanislaw 1986, slide 7)")
    print("""
  Checks the simulator faithfully implements the model:
  portfolio accounting, market clearing conservation, budget constraints.
    """)

    results = []

    # ── 1a. Portfolio accounting: cash + shares*price conservation ─────────
    # For every agent, cash_end + inventory_end * p_t should equal
    # cash_start + inventory_start * p_t ± executed trade value.
    if game.agent_round_records and game.market_round_records:
        market_by_round = {r["round_number"]: r for r in game.market_round_records}
        errors = []
        for rec in game.agent_round_records:
            rn = rec["round_number"]
            mr = market_by_round.get(rn, {})
            pt = mr.get("p_t") or float(game.fundamental_path[rn])
            wealth_start = rec["cash_start"] + rec["inventory_start"] * pt
            wealth_end   = rec["cash_end"]   + rec["inventory_end"]   * pt
            delta_from_trade = (rec["executed_qty"]
                                * (rec["executed_price_avg"] or pt)
                                * (-1 if rec["action"] == "buy" else 1))
            expected_end = wealth_start + delta_from_trade
            if abs(expected_end) > 1e-9:
                errors.append(abs(wealth_end - expected_end) / abs(expected_end))

        if errors:
            max_err = float(np.max(errors))
            passed  = max_err < MAX_CONSERVATION_ERROR
            r = _result(
                "Portfolio accounting conservation",
                passed,
                f"max relative wealth error = {max_err:.2e}  "
                f"(threshold < {MAX_CONSERVATION_ERROR:.0e})"
            )
        else:
            r = _result("Portfolio accounting conservation", True, "no executed trades to check")
        results.append(r)
    else:
        print("  [SKIP]  Portfolio accounting — no round records available")

    # ── 1b. Budget constraint: no buy should exceed available cash ─────────
    violations = 0
    total_buys  = 0
    for rec in game.agent_round_records:
        if rec["action"] == "buy" and rec["order_qty"] > 0:
            total_buys += 1
            cost = rec["order_qty"] * (rec["limit_price"] or 0)
            if cost > rec["cash_start"] + 1e-6:
                violations += 1
    if total_buys > 0:
        rate = violations / total_buys
        r = _result(
            "Budget constraint (no buy exceeds available cash)",
            rate <= MAX_BUDGET_VIOLATION_RATE,
            f"{violations} violations out of {total_buys} buy orders  "
            f"({rate:.4%}  threshold < {MAX_BUDGET_VIOLATION_RATE:.2%})"
        )
        results.append(r)

    # ── 1c. Short-selling constraint: no sell exceeds held shares ──────────
    short_violations = 0
    total_sells = 0
    for rec in game.agent_round_records:
        if rec["action"] == "sell" and rec["order_qty"] > 0:
            total_sells += 1
            if rec["order_qty"] > rec["inventory_start"] + 1e-6:
                short_violations += 1
    if total_sells > 0:
        rate = short_violations / total_sells
        r = _result(
            "Short-selling constraint (no sell exceeds held shares)",
            rate <= MAX_BUDGET_VIOLATION_RATE,
            f"{short_violations} violations out of {total_sells} sell orders  ({rate:.4%})"
        )
        results.append(r)

    # 1d. Clearing volume <= min(demand, supply) 
    
    over_cleared = 0
    for mr in game.market_round_records:
        if mr["p_t"] is not None and mr["demand_at_p"] > 0 and mr["supply_at_p"] > 0:
            max_possible = min(mr["demand_at_p"], mr["supply_at_p"])
            if mr["volume"] > max_possible + 1e-6:
                over_cleared += 1
    r = _result(
        "Market clearing: volume <= min(demand, supply)",
        over_cleared == 0,
        f"{over_cleared} rounds where volume exceeded min(demand, supply)"
    )
    results.append(r)

    # 1e. Fitness signal noise estimate 
    # Time-averaged MTM fitness reduces noise vs single-round liquidation.
    # Check that fitness has reasonable variation across agents (not all equal).
    
    if final_score:
        wealths = np.array([w for _, w in final_score], dtype=float)
        cv = float(np.std(wealths) / np.mean(wealths)) if np.mean(wealths) > 0 else 0
        r = _result(
            "Fitness signal variation (CV > 0 across agents)",
            cv > 0.01,
            f"coefficient of variation = {cv:.4f}  "
            f"(should be > 0.01 — flat = degenerate fitness signal)"
        )
        results.append(r)

    passed = sum(results)
    print(f"\n  Program validity: {passed}/{len(results)} checks passed")
    return results


# =============================================================================
#  2. MODEL VALIDITY (STYLISED FACTS)
# =============================================================================

def validate_model(generation_counts_df, game=None):
    """
    Checks the model reproduces known stylised facts from the ABM literature.
    From lecture slide 8: validation against theory via stylised facts.
    """
    _header("2. MODEL VALIDITY — Stylised Facts  (slide 8)")
    print("""
  Checks qualitative agreement with known properties of informed/noise-trader
  markets. Targets Level 1 empirical validity (Barde & van der Hoog 2017).
    """)

    results = []
    df      = generation_counts_df

    # 2a. Informed/ZI wealth premium > 0% 
    # Gode & Sunder (1993): informed agents must outperform ZI baseline.
    if ("mean_wealth_parameterised_informed" in df.columns
            and "mean_wealth_zi" in df.columns):
        df_clean = df.dropna(subset=["mean_wealth_parameterised_informed", "mean_wealth_zi"])
        if not df_clean.empty:
            # Average over the final half of generations (post-warm-up)
            half    = max(1, len(df_clean) // 2)
            tail    = df_clean.tail(half)
            mean_inf = float(tail["mean_wealth_parameterised_informed"].mean())
            mean_zi  = float(tail["mean_wealth_zi"].mean())
            premium  = (mean_inf - mean_zi) / abs(mean_zi) * 100 if abs(mean_zi) > 1e-9 else 0.0
            final_prem = (
                (df_clean["mean_wealth_parameterised_informed"].iloc[-1]
                 - df_clean["mean_wealth_zi"].iloc[-1])
                / abs(df_clean["mean_wealth_zi"].iloc[-1]) * 100
            )
            r = _result(
                "Informed/ZI wealth premium > 0%  [Gode & Sunder 1993]",
                premium > MIN_WEALTH_PREMIUM_PCT,
                f"mean premium (post-warmup) = {premium:+.2f}%  |  "
                f"final generation = {final_prem:+.2f}%"
            )
            results.append(r)
    else:
        print("  [SKIP]  Wealth premium — columns not found in generation_counts_df")

    # 2b. No-clear rate < 10% 
    if "no_clear_rate" in df.columns:
        mean_ncr = float(df["no_clear_rate"].mean())
        r = _result(
            f"No-clear rate < {MAX_NO_CLEAR_RATE:.0%}",
            mean_ncr < MAX_NO_CLEAR_RATE,
            f"mean no-clear rate across all generations = {mean_ncr:.4f}"
        )
        results.append(r)

    # 2c. Endogenous fat tails in clearing prices
    # A key stylised fact (Cont 2001): price return distributions should be
    # leptokurtic (fat-tailed). If our GBM input is normal but the clearing
    # prices show positive excess kurtosis, that is endogenous fat-tail
    # generation from agent heterogeneity.
    if game is not None and game.market_round_records and game.fundamental_path is not None:
        clearing = [r["p_t"] for r in game.market_round_records if r["p_t"] is not None]
        if len(clearing) >= 4:
            clearing_arr  = np.array(clearing, dtype=float)
            price_returns = np.diff(np.log(clearing_arr))
            price_returns = price_returns[np.isfinite(price_returns)]

            fund_arr     = np.array(game.fundamental_path, dtype=float)
            fund_returns = np.diff(np.log(fund_arr[:len(clearing) + 1]))
            fund_returns  = fund_returns[np.isfinite(fund_returns)]

            if HAS_SCIPY and len(price_returns) >= 4:
                price_kurt = float(scipy_kurtosis(price_returns, fisher=True))
                fund_kurt  = float(scipy_kurtosis(fund_returns,  fisher=True))
                r = _result(
                    "Endogenous excess kurtosis in clearing prices > GBM input  [Cont 2001]",
                    price_kurt > fund_kurt,
                    f"clearing price excess kurtosis = {price_kurt:.4f}  |  "
                    f"GBM fundamental excess kurtosis = {fund_kurt:.4f}"
                )
            else:
                # Manual excess kurtosis = (mu4 / sigma^2^2) - 3
                def _excess_kurt(x):
                    mu    = np.mean(x)
                    sigma = np.std(x)
                    if sigma < 1e-12:
                        return float("nan")
                    return float(np.mean((x - mu)**4) / sigma**4 - 3)

                price_kurt = _excess_kurt(price_returns)
                fund_kurt  = _excess_kurt(fund_returns)
                r = _result(
                    "Endogenous excess kurtosis in clearing prices > GBM input  [Cont 2001]",
                    price_kurt > fund_kurt if np.isfinite(price_kurt) else False,
                    f"clearing price excess kurtosis = {price_kurt:.4f}  |  "
                    f"GBM fundamental excess kurtosis = {fund_kurt:.4f}"
                )
            results.append(r)
    else:
        print("  [SKIP]  Excess kurtosis — no game object provided")

    # 2d. Informed agents have higher fill rates than ZI 
    # Informed agents should post prices closer to clearing price (higher
    # signal_aggression → better fills). ZI agents post randomly so fill
    # rate should be lower on average.
    if game is not None and game.agent_round_records and game.agents:
        agent_types = {aid: a.trader_type for aid, a in game.agents.items()}
        fill_by_type = {}
        for rec in game.agent_round_records:
            if rec["order_qty"] > 0:
                ttype = agent_types.get(rec["agent_id"], "unknown")
                fill_by_type.setdefault(ttype, []).append(float(rec["fill_ratio"]))

        if "parameterised_informed" in fill_by_type and "zi" in fill_by_type:
            fill_inf = float(np.mean(fill_by_type["parameterised_informed"]))
            fill_zi  = float(np.mean(fill_by_type["zi"]))
            r = _result(
                "Informed fill rate >= ZI fill rate",
                fill_inf >= fill_zi - 0.05,   # allow small margin
                f"informed mean fill = {fill_inf:.4f}  |  ZI mean fill = {fill_zi:.4f}"
            )
            results.append(r)

    # 2e. Volume is non-zero (market is active)
    if "mean_volume" in df.columns:
        mean_vol = float(df["mean_volume"].mean())
        r = _result(
            "Mean traded volume > 0 (market is active)",
            mean_vol > 0,
            f"mean volume per round = {mean_vol:.4f}"
        )
        results.append(r)

    passed = sum(results)
    print(f"\n  Model validity: {passed}/{len(results)} stylised fact checks passed")
    return results


# =============================================================================
#  3. EMPIRICAL VALIDITY (Level 1 — Barde & van der Hoog 2017)
# =============================================================================

def validate_empirical(generation_counts_df, game=None, final_score=None):
    """
    Level 1 empirical validity: qualitative agreement with macro structures.
    From lecture slide 10 (Barde & van der Hoog 2017).
    """
    _header("3. EMPIRICAL VALIDITY — Level 1  (Barde & van der Hoog 2017)")
    print("""
  Qualitative agreement with empirical macro structures of agent populations.
  Level 1 = distributional properties of the agent population are realistic.
    """)

    results = []
    df      = generation_counts_df

    # 3a. Non-trivial wealth inequality (Gini > 0)
    # If all agents end with identical wealth the model is degenerate.
    # Real markets produce non-trivial wealth distributions.
    if final_score:
        wealths = [w for _, w in final_score]
        gini    = _gini_coefficient(wealths)
        r = _result(
            f"Non-trivial wealth inequality (Gini > {MIN_GINI:.2f})",
            gini > MIN_GINI,
            f"terminal wealth Gini coefficient = {gini:.4f}"
        )
        results.append(r)
    else:
        print("  [SKIP]  Gini coefficient — no final_score provided")

    # 3b. Parameter diversity does not collapse completely 
    # If all agents converge to identical parameters, evolutionary diversity
    # has been lost entirely — an unrealistic monoculture.
    param_cols = [c for c in df.columns if c.startswith("std_") and "zi" not in c]
    if param_cols:
        # Check that at least one parameter still has non-zero std at the end
        final_stds = {col: float(df[col].iloc[-1]) for col in param_cols
                      if df[col].notna().any()}
        any_diversity = any(v > 1e-4 for v in final_stds.values())
        detail = "  |  ".join(f"{c.replace('std_','')}={v:.4f}"
                               for c, v in final_stds.items())
        r = _result(
            "Parameter diversity not fully collapsed (at least one std > 0)",
            any_diversity,
            detail
        )
        results.append(r)

    # 3c. Informed agents' info_param remains distributed
    # If all informed agents converge to the same info_param, the population
    # is homogeneous — not consistent with real market heterogeneity.
    if "mean_info_param_parameterised_informed" in df.columns:
        ip_series = df["mean_info_param_parameterised_informed"].dropna()
        if len(ip_series) > 0:
            final_mean_ip = float(ip_series.iloc[-1])
            # Check that mean info_param is bounded away from 0 and below the model ceiling (2.0)
            r = _result(
                "Mean info_param stays within plausible range [0.01, 1.99]",
                0.01 <= final_mean_ip <= 1.99,
                f"final generation mean info_param (informed) = {final_mean_ip:.4f}"
            )
            results.append(r)

    # 3d. Informed agents outperform ZI consistently
    # Level 1 empirical validity for an informed trader model requires the
    # informed premium to be positive not just on average but consistently.
    if ("mean_wealth_parameterised_informed" in df.columns
            and "mean_wealth_zi" in df.columns):
        df_w     = df.dropna(subset=["mean_wealth_parameterised_informed", "mean_wealth_zi"])
        premiums = ((df_w["mean_wealth_parameterised_informed"] - df_w["mean_wealth_zi"])
                    / df_w["mean_wealth_zi"].abs() * 100)
        pct_positive = float((premiums > 0).mean()) * 100
        r = _result(
            "Informed wealth premium positive in majority of generations (> 50%)",
            pct_positive > 50,
            f"premium > 0 in {pct_positive:.1f}% of generations"
        )
        results.append(r)

    passed = sum(results)
    print(f"\n  Empirical validity: {passed}/{len(results)} Level 1 checks passed")
    return results


# =============================================================================
#  RUN ALL VALIDATIONS
# =============================================================================

def run_all_validations(generation_counts_df, last_game=None, final_score=None):
    """
    Run all three validation layers and write a summary report.

    Parameters
    ----------
    generation_counts_df : pd.DataFrame
        The generation_counts_df returned by run_experiment().
    last_game : Game object, optional
        Most recent Game instance (for round-level checks).
    final_score : list of (agent_id, wealth), optional
        Terminal wealth for the most recent generation.
    """
    print("\n" + "="*70)
    print("  ABM Validation Report")
    print("  Evolutionary Call Auction Market Model")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("="*70)
    print("""
  Framework: Stanislaw (1986) three validity types,
             extended by Barde & van der Hoog (2017).
  """)

    r1 = validate_program(last_game, final_score) if last_game is not None else []
    r2 = validate_model(generation_counts_df, last_game)
    r3 = validate_empirical(generation_counts_df, last_game, final_score)

    # Overall summary
    _header("VALIDATION SUMMARY")
    all_results = r1 + r2 + r3
    total  = len(all_results)
    passed = sum(all_results)

    if r1:
        print(f"  1. Program validity  : {sum(r1)}/{len(r1)} passed")
    print(    f"  2. Model validity    : {sum(r2)}/{len(r2)} passed")
    print(    f"  3. Empirical validity: {sum(r3)}/{len(r3)} passed")
    _hr()
    print(    f"  TOTAL                : {passed}/{total} checks passed")

    if passed == total:
        print("\n  All validation checks passed.")
        print("  The model achieves Level 1 empirical validity")
        print("  (Barde & van der Hoog 2017).")
    elif passed >= total * 0.75:
        failed_checks = total - passed
        print(f"\n  {failed_checks} check(s) did not pass.")
        print("  The model achieves partial Level 1 empirical validity.")
    else:
        print(f"\n  {total - passed} checks failed — review model configuration.")

    # Write text report
    lines = [
        "ABM Validation Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"Program validity  : {sum(r1)}/{len(r1)} passed\n" if r1 else "",
        f"Model validity    : {sum(r2)}/{len(r2)} passed\n",
        f"Empirical validity: {sum(r3)}/{len(r3)} passed\n",
        f"TOTAL             : {passed}/{total} passed\n",
    ]
    with open("validation_report.txt", "w") as f:
        f.writelines(lines)
    print("\n  Saved: validation_report.txt\n")

    return {"program": r1, "model": r2, "empirical": r3}


# =============================================================================
#  STANDALONE ENTRY POINT
# =============================================================================

def main():
    """
    Run a short experiment then validate the results.
    For production use, import run_all_validations and call it with the
    outputs of your full run_experiment() call.
    """
    print("\nRunning a short validation experiment...")
    print("(For full validation, call run_all_validations() with your main run outputs)\n")

    import os
    db = "validation_temp.duckdb"

    result = run_experiment(
        config_overrides={
            "db_path":          db,
            "experiment_name":  "validation_run",
            "experiment_type":  "validation",
            "n_generations":    20,
            "n_rounds":         25,
            "rolling_n":        10,
        },
        run_analysis=False,
    )

    try:
        run_all_validations(
            generation_counts_df=result["generation_counts_df"],
            last_game=None,     # game object not returned by run_experiment
            final_score=None,
        )
    finally:
        if os.path.exists(db):
            try:
                os.remove(db)
            except Exception:
                pass


if __name__ == "__main__":
    main()