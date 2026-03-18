import numpy as np

# ----------------------------
# Noise + signal helpers
# ----------------------------

def assign_noise_parameter_set(n_agents, noise_parameter_distribution_type, distribution_data, seed=None):
    rng = np.random.default_rng(seed)

    if noise_parameter_distribution_type == "uniform":
        return rng.uniform(distribution_data["low"],
                           distribution_data["high"],
                           n_agents)

    elif noise_parameter_distribution_type == "evenly_spaced":
        return np.linspace(distribution_data["low"],
                           distribution_data["high"],
                           n_agents)

    elif noise_parameter_distribution_type == "bimodal":
        n_a = n_agents // 2
        n_b = n_agents - n_a

        group_a = rng.normal(distribution_data["group_a_mean"],
                             distribution_data["group_a_std"],
                             n_a)

        group_b = rng.normal(distribution_data["group_b_mean"],
                             distribution_data["group_b_std"],
                             n_b)

        return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.01)

    elif noise_parameter_distribution_type == "skewed":
        samples = rng.lognormal(mean=distribution_data["mean"],
                                sigma=distribution_data["sigma"],
                                size=n_agents)
        return np.clip(samples, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown noise_parameter_distribution_type : {noise_parameter_distribution_type}")
