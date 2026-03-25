"""
Experiment orchestration: constructs the population, runs the evolutionary
simulation loop, writes results to DuckDB, and optionally triggers post-run
analysis and plotting.
"""

import random
import subprocess
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ulid

from analysis import (
    analyse_game_results,
    compute_strategy_info_param_stats,
    compute_strategy_mean_wealth,
    compute_strategy_param_stats,
)
from constants import PARAM_BOUNDS, COMPARISON_PARAM_SPECS, WEALTH_INFORMED_COL, WEALTH_ZI_COL
from database_creation import create_database
from sim.evolution import count_strategies, evolve_population, initial_population_from_counts
from sim.gbm import simulate_gbm
from sim.runner import GameExecutionError, play_game
from SQL_Functions import AsyncDuckDBWriter


DB_PATH = "experiment_results.duckdb"

DEFAULT_STRATEGY_PARAMS = {
    "qty_aggression": 0.5,
    "signal_aggression": 0.5,
}

DEFAULT_EXPERIMENT_CONFIG = {
    "db_path": DB_PATH,
    "experiment_name": "Experiment Name",
    "experiment_type": "A sub section for the experiment",
    "run_notes": "Notes for the run",
    "experiment_seed": 587756769879879879879,
    "n_zi_agents": 65,
    "n_parameterised_agents": 35,
    "n_generations": 200,
    "n_rounds": 200,
    "total_initial_cash": 1000,
    "total_initial_shares": 10,
    "GBM_S0": 100,
    "GBM_volatility": 0.20,
    "GBM_drift": 0.05,
    "info_param_distribution_type": "evenly_spaced",
    "distribution_data": {"low": 0.0, "high": 2.0},
    "signal_generator_noise_distribution": "lognormal",
    "algorithm_name": "truncation",
    "algorithm_params": {
        "top_n_fraction": 0.3,
        "bottom_k_fraction": 0.3,
        "mutation_rate": 0.05,
        "info_param_mutation_std": 0.01,
        "info_param_bounds": (0.0, 2.0),
        "param_bounds": PARAM_BOUNDS,
        "default_strategy_params": DEFAULT_STRATEGY_PARAMS,
        "frozen_params": set(),
        "crossover_rate": 0.0,
    },
    "rolling_n": 10,
}


def _generate_ulid() -> str:
    """Generate a new ULID string for use as an experiment identifier."""
    return str(ulid.new())


def _py_version() -> str:
    """Return the current Python interpreter version string."""
    import sys

    return str(sys.version)


def _utc_now() -> str:
    """Return the current UTC time formatted as a ISO-8601-style string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _code_version() -> str:
    """Return the git-derived version tag, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _merged_config(overrides=None):
    """Return a deep copy of DEFAULT_EXPERIMENT_CONFIG with any provided overrides applied."""
    config = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    if overrides:
        for key, value in overrides.items():
            config[key] = deepcopy(value)
    return config


def _quantile_or_nan(values, q):
    """Return the q-th quantile of values, or NaN if the list is empty."""
    if not values:
        return np.nan
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _build_market_summary_records(game):
    """
    Build per-round market summary records from a completed game.

    - Extracts bid/ask extremes, quantiles, fundamental price, and mid-price for each round
    - Returns a list of dicts, one per round
    """
    records = []
    market_by_round = _market_by_round_map(game)

    for round_number in range(int(game.n_rounds)):
        orders = game.order_history.get(round_number, [])
        buy_prices = [order["price"] for order in orders if order["action"] == "buy"]
        sell_prices = [order["price"] for order in orders if order["action"] == "sell"]
        market = market_by_round.get(round_number, {})
        best_bid = market.get("best_bid")
        best_ask = market.get("best_ask")
        mid_price = (
            (best_bid + best_ask) / 2.0
            if best_bid is not None and best_ask is not None
            else np.nan
        )
        records.append(
            {
                "round_number": round_number,
                "max_bid": float(max(buy_prices)) if buy_prices else np.nan,
                "max_sell": float(max(sell_prices)) if sell_prices else np.nan,
                "min_bid": float(min(buy_prices)) if buy_prices else np.nan,
                "min_sell": float(min(sell_prices)) if sell_prices else np.nan,
                "bid_price_q2": _quantile_or_nan(buy_prices, 0.5),
                "ask_price_q3": _quantile_or_nan(sell_prices, 0.75),
                "fundamental_price": float(game.fundamental_path[round_number]),
                "mid_price": float(mid_price) if not np.isnan(mid_price) else np.nan,
            }
        )

    return records


def _agent_types_map(game) -> dict:
    """Return a mapping of agent_id to trader_type string for all agents in a game."""
    return {agent_id: agent.trader_type for agent_id, agent in game.agents.items()}


def _market_by_round_map(game) -> dict:
    """Index market_round_records by round_number for O(1) lookup."""
    return {int(r["round_number"]): r for r in game.market_round_records}


def _groupby_strategy_round_mean(rows: list, value_col: str, output_col: str) -> list:
    """Group rows by (round_number, strategy_type) and return the mean of value_col as output_col."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["round_number", "strategy_type"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: output_col})
        .sort_values(["round_number", "strategy_type"])
    )
    return grouped.to_dict("records")


def _build_strategy_profit_records(game):
    """Compute mean per-round profit/loss by strategy type from agent round records."""
    agent_types = _agent_types_map(game)
    rows = []
    for record in game.agent_round_records:
        executed_price = record["executed_price_avg"] if record["executed_price_avg"] is not None else 0.0
        profit_loss = (
            record["cash_end"] + record["inventory_end"] * executed_price
            - record["cash_start"] - record["inventory_start"] * executed_price
        )
        rows.append({
            "round_number": int(record["round_number"]),
            "strategy_type": agent_types.get(record["agent_id"], "unknown"),
            "profit_loss": float(profit_loss),
        })
    return _groupby_strategy_round_mean(rows, "profit_loss", "avg_profit_loss")


def _build_volume_share_records(game):
    """Compute mean per-round volume share by strategy type, expressed as a fraction of total market volume."""
    agent_types = _agent_types_map(game)
    market_volume = {
        int(r["round_number"]): float(r["volume"]) for r in game.market_round_records
    }
    rows = []
    for record in game.agent_round_records:
        round_number = int(record["round_number"])
        volume = market_volume.get(round_number, 0.0)
        rows.append({
            "round_number": round_number,
            "strategy_type": agent_types.get(record["agent_id"], "unknown"),
            "volume_share": (float(record["executed_qty"]) / volume) if volume > 0 else 0.0,
        })
    return _groupby_strategy_round_mean(rows, "volume_share", "avg_volume_share")


def _build_strategy_performance_round_records(game):
    """
    Build per-round, per-strategy aggregated performance metrics from agent round records.

    - Computes wealth, profit/loss, fill rate, aggressiveness, signal accuracy,
      inventory turnover, execution price deviation, volume share, trade size,
      and inventory risk for each (round_number, strategy_type) pair
    - Returns a list of dicts suitable for database insertion or GUI display
    """
    agent_types = _agent_types_map(game)
    market_by_round = _market_by_round_map(game)
    rows = []
    for record in game.agent_round_records:
        round_number = int(record["round_number"])
        market = market_by_round.get(round_number, {})
        market_price = market.get("p_t")
        if market_price is None and game.fundamental_path is not None and round_number < len(game.fundamental_path):
            market_price = float(game.fundamental_path[round_number])
        profit_loss = (
            record["cash_end"] + record["inventory_end"] * float(market_price)
            - record["cash_start"] - record["inventory_start"] * float(market_price)
        ) if market_price is not None else np.nan
        fill_rate = (
            float(record["executed_qty"]) / float(record["order_qty"])
            if float(record["order_qty"]) > 0
            else np.nan
        )
        avg_abs_inventory = (
            abs(float(record["inventory_start"])) + abs(float(record["inventory_end"]))
        ) / 2.0
        inventory_turnover = (
            abs(float(record["inventory_end"]) - float(record["inventory_start"])) / avg_abs_inventory
            if avg_abs_inventory != 0
            else np.nan
        )
        wealth = (
            float(record["cash_end"]) + float(record["inventory_end"]) * float(market_price)
            if market_price is not None
            else np.nan
        )
        execution_price_deviation = (
            (float(record["executed_price_avg"]) - float(market_price)) / abs(float(market_price))
            if record["executed_price_avg"] is not None and market_price is not None and abs(float(market_price)) > 1e-12
            else np.nan
        )
        market_volume = float(market.get("volume", 0.0))
        volume_share = float(record["executed_qty"]) / market_volume if market_volume > 0 else 0.0
        rows.append(
            {
                "round_number": round_number,
                "strategy_type": agent_types.get(record["agent_id"], "unknown"),
                "wealth": wealth,
                "profit_loss": float(profit_loss),
                "fill_rate": fill_rate,
                "aggressiveness": float(record["aggressiveness"]),
                "signal_accuracy": abs(float(record["signal_error"])),
                "inventory_turnover": inventory_turnover,
                "execution_price_deviation": execution_price_deviation,
                "volume_share": volume_share,
                "trade_size": float(record["executed_qty"]),
                "inventory_risk": abs(float(record["inventory_end"])),
            }
        )

    if not rows:
        return []

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["round_number", "strategy_type"], as_index=False)
        .agg(
            num_agents=("strategy_type", "size"),
            avg_wealth=("wealth", "mean"),
            avg_profit_loss=("profit_loss", "mean"),
            avg_fill_rate=("fill_rate", "mean"),
            avg_aggressiveness=("aggressiveness", "mean"),
            avg_signal_accuracy=("signal_accuracy", "mean"),
            avg_inventory_turnover=("inventory_turnover", "mean"),
            avg_execution_price_deviation=("execution_price_deviation", "mean"),
            avg_volume_share=("volume_share", "mean"),
            avg_trade_size=("trade_size", "mean"),
            avg_inventory_risk=("inventory_risk", "mean"),
        )
        .sort_values(["round_number", "strategy_type"])
    )
    return grouped.to_dict("records")


def _build_strategy_performance_generation_records(round_records, generation_id):
    """Aggregate per-round strategy performance records into a single summary row per strategy for generation_id."""
    if not round_records:
        return []

    df = pd.DataFrame(round_records)
    grouped = (
        df.groupby("strategy_type", as_index=False)
        .agg(
            total_agent_rounds=("num_agents", "sum"),
            avg_profit_loss_per_gen=("avg_profit_loss", "mean"),
            avg_fill_rate_per_gen=("avg_fill_rate", "mean"),
            avg_aggressiveness_per_gen=("avg_aggressiveness", "mean"),
            avg_signal_accuracy_per_gen=("avg_signal_accuracy", "mean"),
            avg_inventory_turnover_per_gen=("avg_inventory_turnover", "mean"),
            avg_execution_price_deviation_per_gen=("avg_execution_price_deviation", "mean"),
            avg_volume_share_per_gen=("avg_volume_share", "mean"),
            avg_trade_size_per_gen=("avg_trade_size", "mean"),
            avg_inventory_risk_per_gen=("avg_inventory_risk", "mean"),
        )
        .sort_values("strategy_type")
    )
    grouped.insert(0, "generation_id", int(generation_id))
    return grouped.to_dict("records")


def run_experiment(config_overrides=None, progress_callback=None, run_analysis=True, disable_db_writes=False):
    """
    Run a full evolutionary market experiment.
    """
    config = _merged_config(config_overrides)
    
    # OPTIMIZATION 1: Only setup the database if we actually want to write to it
    if not disable_db_writes:
        create_database(config["db_path"])
        writer = AsyncDuckDBWriter(config["db_path"])
    else:
        writer = None

    strategy_counts = {
        "zi": int(config["n_zi_agents"]),
        "parameterised_informed": int(config["n_parameterised_agents"]),
    }
    n_agents = sum(strategy_counts.values())
    rng = random.Random(config["experiment_seed"])
    population_spec = initial_population_from_counts(
        strategy_counts,
        DEFAULT_STRATEGY_PARAMS,
        param_bounds=PARAM_BOUNDS,
        rng=rng,
    )

    generation_counts = []
    last_game = None
    last_final_score = None
    recent_games = deque(maxlen=int(config["rolling_n"]))
    recent_scores = deque(maxlen=int(config["rolling_n"]))
    py_vers = _py_version()
    code_vers = _code_version()

    try:
        experiment_id = _generate_ulid()
        experiment_creation_time = _utc_now()

        if writer:
            writer.submit(
                "insert_experiment_row",
                experiment_id, config["experiment_name"], config["experiment_type"],
                experiment_creation_time, None, config["run_notes"], int(config["n_generations"]),
                int(config["n_rounds"]), "GBM", py_vers, code_vers, n_agents,
                float(config["total_initial_cash"]), int(config["total_initial_shares"]),
                "call_auction", "maximum_volume_minimum_imbalance", "proportional_rationing",
                "previous_price_proximity", 0.0, config["info_param_distribution_type"],
                config["distribution_data"], config["signal_generator_noise_distribution"],
                config["algorithm_name"], config["algorithm_params"], 0.0,
            )

        for generation in range(int(config["n_generations"])):
            generation_id = generation + 1
            creation_time = _utc_now()

            current_counts = count_strategies(population_spec)
            generation_counts.append({"generation": generation, **current_counts})

            informed_specs = [
                agent for agent in population_spec if agent["trader_type"] == "parameterised_informed"
            ]
            if informed_specs:
                param_means = {
                    param: sum(agent["strategy_params"][param] for agent in informed_specs) / len(informed_specs)
                    for param in ["qty_aggression", "signal_aggression"]
                }
                param_summary = (
                    f"qty_agg={param_means['qty_aggression']:.2f} "
                    f"sig_agg={param_means['signal_aggression']:.2f}"
                )
            else:
                param_summary = "no informed agents"

            print(
                f"Generation {generation} | Generation ID {generation_id} | "
                f"ZI: {current_counts['zi']} | "
                f"Informed: {current_counts['parameterised_informed']} | "
                f"{param_summary}"
            )

            if progress_callback is not None:
                progress_callback({
                    "event": "generation_started",
                    "experiment_id": experiment_id,
                    "generation_id": generation_id,
                    "generation_index": generation,
                    "n_generations": int(config["n_generations"]),
                })

            if writer:
                writer.submit(
                    "insert_generation_row",
                    experiment_id, generation_id, creation_time, None, "STARTED", 0.0,
                )

            seed = f"gbm_generation_{generation}_seed_{config['experiment_seed']}"
            fundamental_path = simulate_gbm(
                float(config["GBM_S0"]), float(config["GBM_volatility"]),
                float(config["GBM_drift"]), int(config["n_rounds"]), seed,
            )

            try:
                final_score, game = play_game(
                    population_spec, int(config["n_rounds"]), int(config["total_initial_shares"]),
                    float(config["total_initial_cash"]), experiment_id, generation_id,
                    config["info_param_distribution_type"], config["distribution_data"],
                    config["signal_generator_noise_distribution"], S0=float(config["GBM_S0"]),
                    gbm_volatility=float(config["GBM_volatility"]), fundamental_path=fundamental_path,
                    seed=seed,
                )
            except GameExecutionError as exc:
                if writer:
                    failed_progress = (exc.current_round / int(config["n_rounds"])) * 100.0 if int(config["n_rounds"]) > 0 else 0.0
                    writer.submit(
                        "update_generation_progress", experiment_id, generation_id,
                        failed_progress, generation_status="FAILED",
                    )
                raise exc.__cause__ if exc.__cause__ is not None else exc
            except Exception:
                if writer:
                    writer.submit(
                        "update_generation_progress", experiment_id, generation_id,
                        0.0, generation_status="FAILED",
                    )
                raise

            last_game = game
            last_final_score = final_score
            recent_games.append(game)
            recent_scores.append(final_score)

            gen_entry = generation_counts[-1]
            for strategy, wealth in compute_strategy_mean_wealth(final_score, game.agents).items():
                gen_entry[f"mean_wealth_{strategy}"] = wealth
            for strategy, stats in compute_strategy_info_param_stats(game.agents).items():
                gen_entry[f"mean_info_param_{strategy}"] = stats["mean"]
                gen_entry[f"std_info_param_{strategy}"]  = stats["std"]
            for param, stats in compute_strategy_param_stats(game.agents).items():
                gen_entry[f"mean_{param}"] = stats["mean"]
                gen_entry[f"std_{param}"] = stats["std"]

            market_records = game.market_round_records
            if market_records:
                gen_entry["mean_volume"] = sum(record["volume"] for record in market_records) / len(market_records)
                gen_entry["no_clear_rate"] = (
                    sum(1 for record in market_records if record["p_t"] is None) / len(market_records)
                )
            else:
                gen_entry["mean_volume"] = float("nan")
                gen_entry["no_clear_rate"] = float("nan")

            if writer:
                writer.submit(
                    "persist_generation_bundle",
                    experiment_id, generation_id, float(config["GBM_S0"]),
                    float(config["GBM_volatility"]), float(config["GBM_drift"]), seed,
                    tuple(enumerate(fundamental_path)), game.agents, game.market_round_records,
                    game.agent_round_records, game.trade_execution_records, _utc_now(),
                    gen_entry.get("mean_qty_aggression"), gen_entry.get("mean_signal_aggression"),
                )

            if progress_callback is not None:
                # OPTIMIZATION 2: Only calculate these heavy dictionary mappings IF we are going to use them!
                strategy_performance_round = _build_strategy_performance_round_records(game)
                strategy_performance_generation = _build_strategy_performance_generation_records(
                    strategy_performance_round, generation_id,
                )
                
                progress_callback({
                    "event": "generation_completed",
                    "experiment_id": experiment_id,
                    "generation_id": generation_id,
                    "generation_index": generation,
                    "n_generations": int(config["n_generations"]),
                    "generation_metrics": {
                        "generation_id": generation_id,
                        "mean_qty_aggression": gen_entry.get("mean_qty_aggression"),
                        "mean_signal_aggression": gen_entry.get("mean_signal_aggression"),
                        "mean_info_param_parameterised_informed": gen_entry.get("mean_info_param_parameterised_informed"),
                        "mean_info_param_zi": gen_entry.get("mean_info_param_zi"),
                        "mean_wealth_parameterised_informed": gen_entry.get("mean_wealth_parameterised_informed"),
                        "mean_wealth_zi": gen_entry.get("mean_wealth_zi"),
                    },
                    "market_history": list(game.market_round_records),
                    "market_summary": _build_market_summary_records(game),
                    "strategy_profit_per_round": _build_strategy_profit_records(game),
                    "volume_share_per_round": _build_volume_share_records(game),
                    "strategy_performance_round": strategy_performance_round,
                    "strategy_performance_generation": strategy_performance_generation,
                })

            if generation < int(config["n_generations"]) - 1:
                evolvable_ids = {aid for aid, agent in game.agents.items() if agent.trader_type != "zi"}
                evolvable_score = [(aid, wealth) for aid, wealth in final_score if aid in evolvable_ids]
                evolvable_agents = {
                    aid: agent for aid, agent in game.agents.items() if aid in evolvable_ids
                }

                evolved = evolve_population(
                    algorithm_name=config["algorithm_name"],
                    final_score=evolvable_score,
                    agents=evolvable_agents,
                    algorithm_params=config["algorithm_params"],
                    rng=rng,
                )

                zi_cohort = [{"trader_type": "zi"} for _ in range(int(config["n_zi_agents"]))]
                population_spec = zi_cohort + evolved
    finally:
        if writer:
            writer.close()

    generation_counts_df = pd.DataFrame(generation_counts)
    print("\nGeneration strategy counts:")
    print(generation_counts_df)

    results_df = None
    if run_analysis and last_game is not None and last_final_score is not None:
        results_df = analyse_game_results(
            last_game, last_final_score, title_prefix=f"{config['experiment_name']} | ",
            generation_counts_df=generation_counts_df, rolling_games=list(recent_games),
            rolling_scores=list(recent_scores),
        )

    if progress_callback is not None:
        progress_callback({
            "event": "experiment_completed",
            "experiment_id": experiment_id,
            "n_generations": int(config["n_generations"]),
        })

    return {
        "experiment_id": experiment_id,
        "generation_counts_df": generation_counts_df,
        "results_df": results_df,
        "config": config,
    }


if __name__ == "__main__":
    run_experiment()
