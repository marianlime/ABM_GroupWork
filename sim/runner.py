from .game import Game


class GameExecutionError(RuntimeError):
    def __init__(self, current_round: int, n_rounds: int):
        super().__init__(f"Game failed at round {current_round} of {n_rounds}")
        self.current_round = current_round
        self.n_rounds = n_rounds


def play_game(
    # --- Population ---
    population_spec: list[dict],
    # --- Generation / Market ---
    n_rounds: int,
    total_initial_shares: float,
    total_initial_cash: float,
    experiment_id: str,
    generation_id: int,
    # --- Signal / Noise ---
    info_param_distribution_type: str,
    distribution_data: dict,
    signal_generator_noise_distribution: str,
    # --- Fundamentals ---
    S0: float | None = None,
    gbm_volatility: float = 0.2,
    fundamental_path=None,
    seed=None
):
    # Create one Game instance for this generation.
    current_game = Game(
        population_spec=population_spec,
        n_rounds=n_rounds,
        total_initial_shares=total_initial_shares,
        total_initial_cash=total_initial_cash,
        experiment_id=experiment_id,
        generation_id=generation_id,
        info_param_distribution_type=info_param_distribution_type,
        distribution_data=distribution_data,
        signal_generator_noise_distribution=signal_generator_noise_distribution,
        S0=S0,
        gbm_volatility=gbm_volatility,
        fundamental_path=fundamental_path,
        seed=seed,
    )

    try:
        # Run the market for all rounds in this game.
        while current_game.current_round < n_rounds:
            _, _, trades = current_game.gather_orders_and_clear(
                current_game.current_round
            )

            # Apply filled trades to each agent's portfolio.
            for trade in trades:
                current_game.update_portfolio(trade)

            current_game.current_round += 1

        # Time-averaged mark-to-market wealth.
        wealth_sum: dict[int, float] = {}
        wealth_count: dict[int, int] = {}
        for record in current_game.agent_round_records:
            aid = record["agent_id"]
            fund_price = float(current_game.fundamental_path[record["round_number"]])
            mtm = record["cash_end"] + record["inventory_end"] * fund_price
            wealth_sum[aid] = wealth_sum.get(aid, 0.0) + mtm
            wealth_count[aid] = wealth_count.get(aid, 0) + 1

        final_score = [
            (aid, wealth_sum[aid] / wealth_count[aid])
            for aid in current_game.agents
            if wealth_count.get(aid, 0) > 0
        ]

    except Exception as exc:
        raise GameExecutionError(current_game.current_round, n_rounds) from exc

    return final_score, current_game
