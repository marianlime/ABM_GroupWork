from .game import Game
import hashlib
from SQL_Functions import update_run_progress, insert_agent_population
from datetime import datetime, timezone


def seed_to_int(seed) -> int | None:
    """Accept int/str/None and convert strings into a stable integer seed."""
    if seed is None:
        return None
    if isinstance(seed, int):
        return seed
    if isinstance(seed, str):
        return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    raise TypeError(f"seed must be int|str|None, got {type(seed)}")


def play_game(
    # --- Population ---
    DB_PATH,
    population_spec: list[dict],
    # --- Run / Market ---
    n_rounds: int,
    total_initial_shares: float,
    total_initial_cash: float,
    cash_to_share_ratio: float,
    run_id: str,
    market_mechanism: str,
    pricing_rule: str,
    rationing_rule: str,
    tie_break_rule: str,
    transaction_cost_rate: float,
    # --- Signal / Noise ---
    noise_parameter_distribution_type: str,
    distribution_data: dict,
    signal_generator_noise_distribution: str,
    bias: float,
    # --- Fundamentals ---
    fundamental_source: str,
    # --- GBM Configuration ---
    S0: float | None = None,
    volatility: float | None = None,
    drift: float | None = None,
    # --- Historical Configuration ---
    ticker: str | None = None,
    interval: str | None = None,
    start_date: str | None = None,
    price_col: str | None = None,
    auto_adjust: bool | None = None,
    # --- Stock Path ---
    fundamental_path=None,
    seed=None
):
    # Create one Game instance for this run / generation.
    current_game = Game(
        DB_PATH=DB_PATH,
        population_spec=population_spec,
        n_rounds=n_rounds,
        total_initial_shares=total_initial_shares,
        total_initial_cash=total_initial_cash,
        cash_to_share_ratio=cash_to_share_ratio,
        run_id=run_id,
        market_mechanism=market_mechanism,
        pricing_rule=pricing_rule,
        rationing_rule=rationing_rule,
        tie_break_rule=tie_break_rule,
        transaction_cost_rate=transaction_cost_rate,
        noise_parameter_distribution_type=noise_parameter_distribution_type,
        distribution_data=distribution_data,
        signal_generator_noise_distribution=signal_generator_noise_distribution,
        bias=bias,
        fundamental_source=fundamental_source,
        S0=S0,
        volatility=volatility,
        drift=drift,
        ticker=ticker,
        interval=interval,
        start_date=start_date,
        price_col=price_col,
        auto_adjust=auto_adjust,
        fundamental_path=fundamental_path,
        seed=seed,
    )

    # Persist the initial population state before any rounds are played.
    insert_agent_population(DB_PATH, run_id, current_game.agents)

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
            run_progress = (current_game.current_round / n_rounds) * 100.0
            update_run_progress(DB_PATH, run_id, run_progress, run_status="RUNNING")

        # Compute terminal wealth for each agent after the final round.
        final_score = []
        for agent_id in current_game.agents:
            final_score.append((agent_id, current_game.liquidate_assets(agent_id)))

    except Exception:
        failed_progress = (current_game.current_round / n_rounds) * 100.0
        update_run_progress(DB_PATH, run_id, run_progress=failed_progress, run_status="FAILED")
        raise

    completion_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    update_run_progress(
        DB_PATH,
        run_id,
        100.0,
        run_status="COMPLETED",
        completion_time=completion_time
    )

    return final_score, current_game