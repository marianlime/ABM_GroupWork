"""This file contains constants to be used across multiple modules"""

# Constants for parameter bounds and comparison specifications used in market simulation analysis/visualisation.
PARAM_BOUNDS = {"qty_aggression":(0.0, 1.0, 0.02), "signal_aggression":(0.0, 1.0, 0.02),}
COMPARISON_PARAM_SPECS = [("mean_qty_aggression","Mean Qty Aggression"),("mean_signal_aggression","Mean Signal Aggression"),("mean_info_param_parameterised_informed","Mean Info Param (informed)"),]

# Constants for column names in final results DataFrames.
WEALTH_INFORMED_COLUMN = "mean_wealth_parameterised_informed"
WEALTH_ZI_COLUMN       = "mean_wealth_zi"
