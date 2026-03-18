from .game import Game
from SQL_Functions import update_run_progress, insert_agent_population
from datetime import datetime, timezone


def play_game(
    # --- Connection ---
    con,
    # --- Population ---
    population_spec: list[dict],
    # --- Run / Market ---
    n_rounds: int,
    total_initial_shares: float,
    total_initial_cash: float,
    run_id: str,
    # --- Signal / Noise ---
    noise_parameter_distribution_type: str,
    distribution_data: dict,
    signal_generator_noise_distribution: str,
    # --- Fundamentals ---
    S0: float | None = None,
    gbm_volatility: float = 0.2,
    fundamental_path=None,
    seed=None
):
    # Create one Game instance for this run / generation.
    current_game = Game(
        population_spec=population_spec,
        n_rounds=n_rounds,
        total_initial_shares=total_initial_shares,
        total_initial_cash=total_initial_cash,
        run_id=run_id,
        noise_parameter_distribution_type=noise_parameter_distribution_type,
        distribution_data=distribution_data,
        signal_generator_noise_distribution=signal_generator_noise_distribution,
        S0=S0,
        gbm_volatility=gbm_volatility,
        fundamental_path=fundamental_path,
        seed=seed,
    )

    # Persist the initial population state before any rounds are played.
    insert_agent_population(con, run_id, current_game.agents)

    try:
        # Run the market for all rounds in this game.
        while current_game.current_round < n_rounds:
            best_price, total_volume, trades = current_game.gather_orders_and_clear(
                current_game.current_round
            )

            # Apply filled trades to each agent's portfolio.
            for trade in trades:
                current_game.update_portfolio(trade)

            current_game.current_round += 1

        # Time-averaged mark-to-market wealth.
        # Averaging over all T rounds (rather than using the single terminal snapshot)
        # reduces the standard error of the fitness signal by ~sqrt(T), making
        # evolutionary selection much less noisy with short generation lengths.
        wealth_sum:   dict[int, float] = {}
        wealth_count: dict[int, int]   = {}
        for record in current_game.agent_round_records:
            aid = record["agent_id"]
            fund_price = float(current_game.fundamental_path[record["round_number"]])
            mtm = record["cash_end"] + record["inventory_end"] * fund_price
            wealth_sum[aid]   = wealth_sum.get(aid, 0.0) + mtm
            wealth_count[aid] = wealth_count.get(aid, 0)  + 1

        final_score = [
            (aid, wealth_sum[aid] / wealth_count[aid])
            for aid in current_game.agents
            if wealth_count.get(aid, 0) > 0
        ]

    except Exception:
        failed_progress = (current_game.current_round / n_rounds) * 100.0
        update_run_progress(con, run_id, run_progress=failed_progress, run_status="FAILED")
        raise

    completion_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    update_run_progress(
        con,
        run_id,
        100.0,
        run_status="COMPLETED",
        completion_time=completion_time
    )

    return final_score, current_game
