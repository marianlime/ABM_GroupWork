"""Shared project defaults and constant field names."""

DB_PATH = "Database/experiment_results.duckdb"

PARAM_BOUNDS = {
    "qty_aggression": (0.0, 1.0, 0.02),
    "signal_aggression": (0.0, 1.0, 0.02),
}

COMPARISON_PARAM_SPECS = [
    ("mean_qty_aggression", "Mean Qty Aggression"),
    ("mean_signal_aggression", "Mean Signal Aggression"),
    ("mean_info_param_parameterised_informed", "Mean Info Param (informed)"),
]

WEALTH_INFORMED_COLUMN = "mean_wealth_parameterised_informed"
WEALTH_ZI_COLUMN = "mean_wealth_zi"

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
