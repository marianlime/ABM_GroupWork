"""
Helpers for sampling agent information-quality parameters from configurable
distributions, supporting uniform, evenly-spaced, bimodal, and skewed modes.
"""

import numpy as np


# ── Noise + signal helpers ────────────────────────────────────────────────────

def assign_info_param_set(n_agents, info_param_distribution_type, distribution_data, seed=None):
    """
    Draw an array of info_param values for n_agents from the specified distribution.

    - distribution_type "uniform"      : samples from U(low, high)
    - distribution_type "evenly_spaced": linspace from low to high
    - distribution_type "bimodal"      : two Gaussian groups clipped to [0.01, 1.01]
    - distribution_type "skewed"       : log-normal samples clipped to [0, 1]
    """
    rng = np.random.default_rng(seed)

    if info_param_distribution_type == "uniform":
        return rng.uniform(distribution_data["low"],
                           distribution_data["high"],
                           n_agents)

    elif info_param_distribution_type == "evenly_spaced":
        return np.linspace(distribution_data["low"],
                           distribution_data["high"],
                           n_agents)

    elif info_param_distribution_type == "bimodal":
        n_a = n_agents // 2
        n_b = n_agents - n_a

        group_a = rng.normal(distribution_data["group_a_mean"],
                             distribution_data["group_a_std"],
                             n_a)

        group_b = rng.normal(distribution_data["group_b_mean"],
                             distribution_data["group_b_std"],
                             n_b)

        return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.01)

    elif info_param_distribution_type == "skewed":
        samples = rng.lognormal(mean=distribution_data["mean"],
                                sigma=distribution_data["sigma"],
                                size=n_agents)
        return np.clip(samples, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown info_param_distribution_type : {info_param_distribution_type}")
