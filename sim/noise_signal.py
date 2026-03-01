import numpy as np

# ----------------------------
# Noise + signal helpers
# ----------------------------

def assign_noise_parameter_set(num_agents, dist_type="uniform"):
    if dist_type == "uniform":
        return np.random.uniform(0.0, 0.5, num_agents)
    elif dist_type == "evenly_spaced":
        return np.linspace(0, 1, num_agents)
    elif dist_type == "bimodal":
        group_a = np.random.normal(0.1, 0.05, num_agents // 2)
        group_b = np.random.normal(0.9, 0.05, num_agents // 2)
        return np.clip(np.concatenate([group_a, group_b]), 0.01, 1.5)
    elif dist_type == "skewed":
        return np.random.lognormal(-1, 0.5, num_agents)

def signal_generator(noise_parameter, S_next, bias = 0.0, noise_distribution='lognormal'):
    if noise_distribution == 'lognormal':
        # multiplicative lognormal noise
        # adding a bias factor into the noise as it can replicate real world market "average" belief(bullish or bearish)
        return S_next * np.exp(np.random.normal(bias, noise_parameter))

    if noise_distribution == 'uniform':
        # multiplicative uniform noise around 1: U(1-σ, 1+σ)
        # clip lower bound so the multiplier stays positive
        sigma = float(noise_parameter)
        low = max(1.0 - sigma, 1e-6)
        high = 1.0 + sigma
        return S_next * np.random.uniform(low, high)

    raise ValueError(f"Unknown noise_distribution: {noise_distribution}")