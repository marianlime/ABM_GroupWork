from sim import play_game
from analysis import analyse_game_results

# ----------------------------
# Example params + run
# ----------------------------

n_strategic_agents = 50
n_zi_agents = 100
n_rounds = 10
S0 = 100
volatility = 0.1
drift = 0
total_initial_shares = 100
cash_to_share_ratio = 1
run_id = 1
noise_param_dist_type = "uniform"
signal_noise_distribution = "lognormal"
seed = None

final_score, g = play_game(
    n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
    total_initial_shares, cash_to_share_ratio, run_id,
    noise_param_dist_type=noise_param_dist_type,
    signal_noise_distribution=signal_noise_distribution,
    seed=seed
)

analyse_game_results(g, final_score, n_strategic_agents=None, title_prefix="")
print("Example (all results):", final_score)
print("Last clearing price:", g.price_history.get(n_rounds - 1))