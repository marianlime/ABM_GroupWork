"""
constants.py
============
Single source of truth for shared constants used across main.py,
analysis.py, gui.py, and comparison_runs.py.
"""

# Bounds for the two learnable strategy parameters: (min, max, mutation_step)
PARAM_BOUNDS = {
    "qty_aggression":    (0.0, 1.0, 0.02),
    "signal_aggression": (0.0, 1.0, 0.02),
}

# Display specs for per-parameter comparison plots: (df_column, human label)
COMPARISON_PARAM_SPECS = [
    ("mean_qty_aggression",                    "Mean Qty Aggression"),
    ("mean_signal_aggression",                 "Mean Signal Aggression"),
    ("mean_info_param_parameterised_informed", "Mean Info Param (informed)"),
]

# Column names for per-generation mean wealth in generation_counts_df
WEALTH_INFORMED_COL = "mean_wealth_parameterised_informed"
WEALTH_ZI_COL       = "mean_wealth_zi"
