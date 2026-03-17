# -------------- Standard Library -------------
import os
import sys
import subprocess
import random
from collections import deque
from datetime import datetime, timezone

# ---------- Third-Party Libraries ------------
import duckdb
import ulid
import pandas as pd

# ------------- Project Modules ---------------
from database_creation import create_database
from analysis import (
    analyse_game_results,
    compute_strategy_mean_wealth,
    compute_strategy_mean_info_param,
    compute_strategy_param_stats,
)
from sim.gbm import simulate_gbm
from sim.runner import play_game
from sim.evolution import (
    initial_population_from_counts,
    evolve_population,
    count_strategies,
)
from SQL_Functions import (
    insert_run_row,
    insert_gbm_config_row,
    insert_fundamental_series,
    insert_agent_round_rows,
    insert_market_round_rows,
)
# ---------------------------------------------


DB_PATH = "testdataset.duckdb"

if not os.path.exists(DB_PATH):
    create_database(DB_PATH)


# ═════════════════════════════════════════════
#  EXPERIMENT PARAMETERS  ← edit here
# ═════════════════════════════════════════════

# --- Experiment metadata ---
experiment_name = "Experiment Name"
experiment_type = "A sub section for the experiment"
run_notes       = "Notes for the run"
experiment_seed = 42

# --- Population ---
n_zi_agents           = 35
n_parameterised_agents = 65

# --- Simulation ---
n_generations = 200
n_rounds      = 10

# --- Initial endowments ---
total_initial_cash   = 1000
total_initial_shares = 10

# --- GBM fundamental path ---
GBM_S0         = 100
GBM_volatility = 0.2
GBM_drift      = 0.01

# --- Noise / signal ---
# Distribution of info_param across agents.
# Types: "uniform", "bimodal", "skewed", "evenly_spaced"
noise_parameter_distribution_type  = "evenly_spaced"
distribution_data                   = {"low": 0.0, "high": 1.0}

# Noise model for individual price signals.  Options: "lognormal", "uniform"
signal_generator_noise_distribution = "lognormal"

# --- Evolution: parameter search space ---
# Each entry: (min, max, gaussian_mutation_std)
PARAM_BOUNDS = {
    "direction_bias": (-1.0,  1.0,  0.10),
    "aggression":     ( 0.1,  5.0,  0.20),
    "patience":       ( 0.0,  1.0,  0.05),
    "threshold":      ( 0.0,  0.50, 0.03),
}

# Starting values for a newly initialised parameterised_informed agent.
DEFAULT_STRATEGY_PARAMS = {
    "direction_bias": 1.0,
    "aggression":     1.0,
    "patience":       1.0,
    "threshold":      0.0,
}

# --- Evolution algorithm ---
algorithm_name   = "truncation"
algorithm_params = {
    "top_n":    10,
    "bottom_k": 10,
    # Probability of a jump mutation (uniform re-sample within bounds).
    # Keep low to favour exploitation over exploration.
    "mutation_rate":           0.05,
    "info_param_mutation_std": 0.01,
    "info_param_bounds":       (0.0, 1.0),
    # Passed through to _make_child — defined above so they're easy to edit.
    "param_bounds":            PARAM_BOUNDS,
    "default_strategy_params": DEFAULT_STRATEGY_PARAMS,
    # Parameters listed here are inherited unchanged from parent to child.
    # e.g. {"info_param"} to fix noise quality; {} to evolve everything.
    "frozen_params":           set(),
}

# --- Rolling analysis window ---
# Post-run plots are averaged over the last rolling_n generations.
rolling_n = 20

# --- Market  ---
# call_auction | max_volume_min_imbalance | pro-rata rationing | price_proximity tie-break

# ═════════════════════════════════════════════
#  DERIVED / INTERNAL  — do not edit below
# ═════════════════════════════════════════════

strategy_counts = {
    "zi":                    n_zi_agents,
    "parameterised_informed": n_parameterised_agents,
}

rng             = random.Random(experiment_seed)
population_spec = initial_population_from_counts(strategy_counts, DEFAULT_STRATEGY_PARAMS, param_bounds=PARAM_BOUNDS, rng=rng)
n_agents        = sum(strategy_counts.values())

generation_counts = []
last_game         = None
last_final_score  = None
_recent_games     = deque(maxlen=rolling_n)
_recent_scores    = deque(maxlen=rolling_n)


def _generate_ulid() -> str:
    return str(ulid.new())


def _py_version() -> str:
    return str(sys.version)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


py_vers   = _py_version()
code_vers = _code_version()

# ----------------------------
# Evolutionary loop — one shared DB connection for the entire run
# ----------------------------

con = duckdb.connect(DB_PATH)
con.execute("BEGIN TRANSACTION")
try:
    for generation in range(n_generations):
        run_id        = _generate_ulid()
        creation_time = _utc_now()

        current_counts = count_strategies(population_spec)
        generation_counts.append({"generation": generation, **current_counts})

        informed_specs = [a for a in population_spec if a["trader_type"] == "parameterised_informed"]
        if informed_specs:
            _pm = {p: sum(a["strategy_params"][p] for a in informed_specs) / len(informed_specs)
                   for p in ["direction_bias", "aggression", "patience", "threshold"]}
            _param_str = (f"dir={_pm['direction_bias']:.2f} agg={_pm['aggression']:.2f} "
                          f"pat={_pm['patience']:.2f} thr={_pm['threshold']:.2f}")
        else:
            _param_str = "no informed agents"
        print(f"Generation {generation} | Run {run_id} | "
              f"ZI: {current_counts['zi']} | Informed: {current_counts['parameterised_informed']} | "
              f"{_param_str}")

        insert_run_row(
            con,
            run_id,
            experiment_name,
            f"{experiment_type} | generation_{generation}",
            creation_time,
            None,           # completion_time — filled in by play_game
            run_notes,
            n_rounds,
            "GBM",
            "STARTED",
            0,
            py_vers,
            code_vers,
            n_agents,
            total_initial_cash,
            total_initial_shares,
            "call_auction",
            "maximum_volume_minimum_imbalance",
            "proportional_rationing",
            "previous_price_proximity",
            0.0,            # transaction_cost_rate
            noise_parameter_distribution_type,
            distribution_data,
            signal_generator_noise_distribution,
            0.0,            # bias (fixed; kept in schema for backwards compatibility)
        )

        # Generate a fresh GBM path for this generation
        seed             = f"gbm_generation_{generation}_seed_{experiment_seed}"
        fundamental_path = simulate_gbm(GBM_S0, GBM_volatility, GBM_drift, n_rounds, seed)

        insert_gbm_config_row(con, run_id, GBM_S0, GBM_volatility, GBM_drift, seed)
        insert_fundamental_series(con, run_id, tuple(enumerate(fundamental_path)))

        # Run this generation's game
        final_score, g = play_game(
            con,
            population_spec,
            n_rounds,
            total_initial_shares,
            total_initial_cash,
            run_id,
            noise_parameter_distribution_type,
            distribution_data,
            signal_generator_noise_distribution,
            S0=GBM_S0,
            fundamental_path=fundamental_path,
            seed=seed,
        )

        insert_market_round_rows(con, g.market_round_records)
        insert_agent_round_rows(con, g.agent_round_records)

        last_game        = g
        last_final_score = final_score
        _recent_games.append(g)
        _recent_scores.append(final_score)

        # Record per-generation metrics
        gen_entry = generation_counts[-1]
        for strategy, wealth in compute_strategy_mean_wealth(final_score, g.agents).items():
            gen_entry[f"mean_wealth_{strategy}"] = wealth
        for strategy, ip in compute_strategy_mean_info_param(g.agents).items():
            gen_entry[f"mean_info_param_{strategy}"] = ip
        for param, stats in compute_strategy_param_stats(g.agents).items():
            gen_entry[f"mean_{param}"] = stats["mean"]
            gen_entry[f"std_{param}"]  = stats["std"]

        _market_records = g.market_round_records
        if _market_records:
            gen_entry["mean_volume"]   = sum(r["volume"] for r in _market_records) / len(_market_records)
            gen_entry["no_clear_rate"] = sum(1 for r in _market_records if r["p_t"] is None) / len(_market_records)
        else:
            gen_entry["mean_volume"]   = float("nan")
            gen_entry["no_clear_rate"] = float("nan")

        # Evolve into the next generation (ZI cohort is fixed)
        if generation < n_generations - 1:
            evolvable_ids    = {aid for aid, agent in g.agents.items() if agent.trader_type != "zi"}
            evolvable_score  = [(aid, w) for aid, w in final_score if aid in evolvable_ids]
            evolvable_agents = {aid: agent for aid, agent in g.agents.items() if aid in evolvable_ids}

            evolved = evolve_population(
                algorithm_name=algorithm_name,
                final_score=evolvable_score,
                agents=evolvable_agents,
                algorithm_params=algorithm_params,
                rng=rng,
            )

            zi_cohort       = [{"trader_type": "zi"} for _ in range(n_zi_agents)]
            population_spec = zi_cohort + evolved

    con.execute("COMMIT")
finally:
    con.close()


# ----------------------------
# Post-run outputs
# ----------------------------

generation_counts_df = pd.DataFrame(generation_counts)
print("\nGeneration strategy counts:")
print(generation_counts_df)

results_df = analyse_game_results(
    last_game,
    last_final_score,
    title_prefix=f"{experiment_name} | ",
    generation_counts_df=generation_counts_df,
    rolling_games=list(_recent_games),
    rolling_scores=list(_recent_scores),
)
