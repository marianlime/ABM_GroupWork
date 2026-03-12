#-------------- Standard Library -------------
import os
import sys
import subprocess
import random
from datetime import datetime, timezone

#---------- Third-Party Libraries ------------
import ulid
import pandas as pd

#------------- Project Modules ---------------
from database_creation import create_database
from analysis import analyse_game_results

from sim.gbm import (
    simulate_gbm,
    simulate_from_yfinance,
)

from sim.runner import play_game
from sim.evolution import (
    initial_population_from_counts,
    evolve_population,
    count_strategies,
)

from SQL_Functions import (
    insert_run_row,
    insert_gbm_config_row,
    insert_hist_config_row,
    insert_fundamental_series,
    insert_agent_population,
    insert_agent_round_rows,
    insert_market_round_rows,
)
#---------------------------------------------


DB_PATH = "testdataset.duckdb"

if not os.path.exists(DB_PATH):
    create_database(DB_PATH)

# INITIAL PARAMETER SETS DO NOT CHANGE
S0 = None
volatility = None
drift = None
seed = None
ticker = None
interval = None
start_date = None
price_col = None
auto_adjust = None
fundamental_path = None
completion_time = None
# INITIAL PARAMETER SETS DO NOT CHANGE


def compute_end_date(start_date, n_rounds, interval):
    start_date = pd.Timestamp(start_date)

    interval_map = {
        "1m": pd.Timedelta(minutes=1),
        "2m": pd.Timedelta(minutes=2),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "60m": pd.Timedelta(hours=1),
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "1wk": pd.Timedelta(weeks=1),
        "1mo": pd.DateOffset(months=1)
    }

    step = interval_map[interval]
    end_date = start_date + n_rounds * step
    return end_date


# ----------------------------
# Example params + run
# ----------------------------

def generate_ULID() -> str:
    return str(ulid.new())


def generate_py_Vers() -> str:
    return str(sys.version)


def generate_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def generate_code_Vers() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# -----------------------------
# EVOLUTION / EXPERIMENT DATA
# -----------------------------

n_generations = 5

algorithm_name = "truncation"
algorithm_params = {
    "top_n": 10,
    "bottom_k": 10,
}

experiment_seed = 42
rng = random.Random(experiment_seed)

# -----------------------------
# RUN DATA
# -----------------------------

n_zi_agents = 20
n_signal_following_agents = 20
n_utility_maximiser_agents = 20
n_contrarian_agents = 20
n_adapt_sig_agents = 20

strategy_counts = {
    "zi": n_zi_agents,
    "signal_following": n_signal_following_agents,
    "utility_maximiser": n_utility_maximiser_agents,
    "contrarian": n_contrarian_agents,
    "adapt_sig": n_adapt_sig_agents
}

# Generation 0 population
population_spec = initial_population_from_counts(strategy_counts)

# Run / experiment info
experiment_name = "Experiment Name"
experiment_type = "A sub section for the experiment"
run_notes = "Notes for the run"
n_rounds = 100
fundamental_source = "GBM"   # ["GBM", "Historical"]

# Version data
py_vers = generate_py_Vers()
code_vers = generate_code_Vers()

# Agent Data
n_agents = sum(strategy_counts.values())
total_initial_cash = 1000
total_initial_shares = 10
cash_to_share_ratio = total_initial_cash / total_initial_shares

# Market Mechanisms
market_mechanism = "call_auction"
pricing_rule = "maximum_volume_minimum_imbalance"
rationing_rule = "proportional_rationing"
tie_break_rule = "previous_price_proximity"
transaction_cost_rate = 0.000

# Noise / signal setup
noise_parameter_distribution_type = "uniform"  # [uniform, bimodal, skewed, evenly_spaced]

if noise_parameter_distribution_type == "uniform":
    low = 0.0
    high = 0.5
    distribution_data = {
        "low": low,
        "high": high
    }

elif noise_parameter_distribution_type == "bimodal":
    group_a_mean = 0.1
    group_a_std = 0.05
    group_b_mean = 0.9
    group_b_std = 0.05
    distribution_data = {
        "group_a_mean": group_a_mean,
        "group_a_std": group_a_std,
        "group_b_mean": group_b_mean,
        "group_b_std": group_b_std
    }

elif noise_parameter_distribution_type == "skewed":
    mean = -1
    sigma = 0.5
    distribution_data = {
        "mean": mean,
        "sigma": sigma
    }

elif noise_parameter_distribution_type == "evenly_spaced":
    low = 0.05
    high = 0.1
    distribution_data = {
        "low": low,
        "high": high
    }

else:
    raise ValueError(
        f"Unknown noise_parameter_distribution_type: {noise_parameter_distribution_type}"
    )

signal_generator_noise_distribution = "lognormal"   # [lognormal, uniform]
bias = 0.0

# Store composition history for plotting / inspection later
generation_counts = []

# Keep references to the last generation's outputs
last_game = None
last_final_score = None

# ----------------------------
# Evolutionary loop
# ----------------------------

for generation in range(n_generations):
    run_id = generate_ULID()
    creation_time = generate_time()
    completion_time = None

    # Each generation is stored as its own run in the DB.
    run_status = "STARTED"
    run_progress = 0

    # Record composition at the start of the generation.
    current_counts = count_strategies(population_spec)
    generation_counts.append({
        "generation": generation,
        **current_counts
    })

    print(f"Generation {generation} | Run {run_id} created | Composition: {current_counts}")

    insert_run_row(
        DB_PATH,
        run_id,
        experiment_name,
        f"{experiment_type} | generation_{generation}",
        creation_time,
        completion_time,
        run_notes,
        n_rounds,
        fundamental_source,
        run_status,
        run_progress,
        py_vers,
        code_vers,
        n_agents,
        total_initial_cash,
        total_initial_shares,
        market_mechanism,
        pricing_rule,
        rationing_rule,
        tie_break_rule,
        transaction_cost_rate,
        noise_parameter_distribution_type,
        distribution_data,
        signal_generator_noise_distribution,
        bias
    )

    # ----------------------------
    # Generate a fresh fundamental path for this generation
    # ----------------------------

    if fundamental_source == "GBM":
        S0 = 100
        volatility = 0.2
        drift = 0.01

        # Use a generation-specific seed so paths differ across generations
        seed = f"gbm_generation_{generation}_seed_{experiment_seed}"

        insert_gbm_config_row(DB_PATH, run_id, S0, volatility, drift, seed)

        fundamental_path = simulate_gbm(S0, volatility, drift, n_rounds, seed)
        fundamental_series = tuple(enumerate(fundamental_path))
        insert_fundamental_series(DB_PATH, run_id, fundamental_series)

        # Reset unused historical fields
        ticker = None
        interval = None
        start_date = None
        price_col = None
        auto_adjust = None

    elif fundamental_source == "Historical":
        ticker = "AAPL"
        interval = "1d"
        start_date = "2024-01-01"
        end_date = compute_end_date(start_date, n_rounds, interval)
        price_col = "Close"
        auto_adjust = True
        seed = None

        insert_hist_config_row(
            DB_PATH,
            run_id,
            ticker,
            interval,
            start_date,
            end_date,
            price_col,
            auto_adjust
        )

        fundamental_path = simulate_from_yfinance(
            ticker,
            start_date,
            n_rounds,
            interval,
            price_col,
            auto_adjust
        )
        fundamental_series = tuple(enumerate(fundamental_path))
        insert_fundamental_series(DB_PATH, run_id, fundamental_series)

        # Reset unused GBM fields
        S0 = None
        volatility = None
        drift = None

    else:
        raise ValueError("Error")

    # ----------------------------
    # Run this generation's game
    # ----------------------------

    final_score, g = play_game(
        DB_PATH,
        population_spec,
        n_rounds,
        total_initial_shares,
        total_initial_cash,
        cash_to_share_ratio,
        run_id,
        market_mechanism,
        pricing_rule,
        rationing_rule,
        tie_break_rule,
        transaction_cost_rate,
        noise_parameter_distribution_type,
        distribution_data,
        signal_generator_noise_distribution,
        bias,
        fundamental_source,
        S0,
        volatility,
        drift,
        ticker,
        interval,
        start_date,
        price_col,
        auto_adjust,
        fundamental_path,
        seed
    )

    # Persist outputs for this generation
    insert_market_round_rows(DB_PATH, g.market_round_records)
    insert_agent_round_rows(DB_PATH, g.agent_round_records)

    # Keep references to the most recent game
    last_game = g
    last_final_score = final_score

    # Evolve into the next generation, unless this is the final generation.
    if generation < n_generations - 1:
        population_spec = evolve_population(
            algorithm_name=algorithm_name,
            final_score=final_score,
            agents=g.agents,
            algorithm_params=algorithm_params,
            rng=rng
        )


# ----------------------------
# Post-run outputs
# ----------------------------

generation_counts_df = pd.DataFrame(generation_counts)
print("\nGeneration strategy counts:")
print(generation_counts_df)

# Analyse the final generation only, to keep changes minimal.
generation_counts_df = pd.DataFrame(generation_counts)

results_df = analyse_game_results(
    last_game,
    last_final_score,
    title_prefix=f"{experiment_name} | Final Generation | ",
    generation_counts_df=generation_counts_df,
)