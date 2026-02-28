from .gbm import simulate_gbm
from .game import game
import numpy as np
import random

# ----------------------------
# Runner
# ----------------------------

def play_game(n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
              total_initial_shares, cash_to_share_ratio, run_id,
              noise_param_dist_type="evenly_spaced", signal_noise_distribution="lognormal",
              seed=None):
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    gbm_path = simulate_gbm(S0, volatility, drift, n_rounds)
    stock_path = gbm_path['stock_path']
    shock_path = gbm_path['shock_path']

    current_game = game(n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift,
                        total_initial_shares, cash_to_share_ratio, run_id, stock_path, shock_path,
                        noise_param_dist_type=noise_param_dist_type,
                        signal_noise_distribution=signal_noise_distribution)

    while current_game.current_round < n_rounds:
        best_price, total_volume, trades = current_game.gather_orders_and_clear(current_game.current_round)
        for trade in trades:
            current_game.update_portfolio(trade)
        current_game.current_round += 1

    final_score = []
    for agent_id in current_game.agents:
        final_score.append((agent_id, current_game.liquidate_assets(agent_id)))

    return final_score, current_game