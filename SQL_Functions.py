import json


def insert_run_row(con,
                   run_id,
                   experiment_name,
                   experiment_type,
                   creation_time,
                   completion_time,
                   run_notes,
                   n_rounds,
                   fundamental_source,
                   run_status,
                   run_progress,
                   py_vers,
                   code_vers,
                   n_agents,
                   total_starting_cash,
                   total_starting_shares,
                   market_mechanism,
                   pricing_rule,
                   rationing_rule,
                   tie_break_rule,
                   transaction_cost_rate,
                   noise_parameter_distribution_type,
                   distribution_data,
                   signal_generator_noise_distribution,
                   bias):
    con.execute("""
        INSERT INTO runs VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, [
        run_id,
        experiment_name,
        experiment_type,
        creation_time,
        completion_time,
        run_notes,
        n_rounds,
        fundamental_source,
        run_status,
        run_progress,
        py_vers,
        code_vers,
        n_agents,
        total_starting_cash,
        total_starting_shares,
        market_mechanism,
        pricing_rule,
        rationing_rule,
        tie_break_rule,
        transaction_cost_rate,
        noise_parameter_distribution_type,
        json.dumps(distribution_data),
        signal_generator_noise_distribution,
        bias
    ])



def insert_gbm_config_row(con, run_id, S0, volatility, drift, seed):
    con.execute(
        """INSERT INTO gbm_config VALUES (?, ?, ?, ?, ?)""",
        [run_id, S0, drift, volatility, seed]
    )


def insert_hist_config_row(con, run_id, ticker, interval, start_date, end_date, price_col, auto_adjust):
    con.execute("""
        INSERT INTO hist_config VALUES (
            ?, ?, ?, ?, ?, ?, ?
        )
    """, [run_id, ticker, interval, start_date, end_date, price_col, auto_adjust])



def insert_fundamental_series(con, run_id, fundamental_series):
    data = []
    for r, p in fundamental_series:
        if isinstance(p, tuple):
            raise TypeError(f"Expected numeric price, got tuple at round {r}: {p}")
        data.append((run_id, int(r), float(p)))

    con.executemany(
        """
        INSERT INTO fundamental_series (run_id, round_number, price)
        VALUES (?, ?, ?)
        """,
        data
    )


def insert_agent_population(con, run_id, agents):
    data = []
    for agent_id, agent in agents.items():
        group_label = "noise" if agent.trader_type == "zi" else "informed"
        data.append((
            run_id,
            int(agent_id),
            str(agent.trader_type),
            float(agent.info_param),
            group_label,
            float(agent.cash),
            float(agent.shares)
        ))

    con.executemany("""INSERT INTO agent_population
                    (run_id, agent_id, strategy_type, noise_parameter, group_label, initial_cash, initial_shares)
                    VALUES (?,?,?,?,?,?,?)""", data)



def insert_agent_round_rows(con, records):
    data = []
    for r in records:
        data.append((
            r["run_id"],
            r["round_number"],
            r["agent_id"],
            r["signal"],
            r["signal_error"],
            r["action"],
            r["limit_price"],
            r["order_qty"],
            r["aggressiveness"],
            r["executed_qty"],
            r["executed_price_avg"],
            r["fill_ratio"],
            r["is_filled"],
            r["is_partial"],
            r["cash_start"],
            r["inventory_start"],
            r["cash_end"],
            r["inventory_end"],
        ))

    con.executemany("""
        INSERT INTO agent_round (
            run_id,
            round_number,
            agent_id,
            signal,
            signal_error,
            action,
            limit_price,
            order_qty,
            aggressiveness,
            executed_qty,
            executed_price_avg,
            fill_ratio,
            is_filled,
            is_partial,
            cash_start,
            inventory_start,
            cash_end,
            inventory_end
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)


def update_run_progress(con, run_id, run_progress, run_status=None, completion_time=None):
    if run_status is not None and completion_time is not None:
        con.execute("""
        UPDATE runs
        SET run_progress = ?, run_status = ?, completion_time = ?
        WHERE run_id = ?
        """, [float(run_progress), run_status, completion_time, run_id])

    elif run_status is not None:
        con.execute("""
            UPDATE runs
            SET run_progress = ?, run_status = ?
            WHERE run_id = ?
        """, [float(run_progress), run_status, run_id])

    else:
        con.execute("""
            UPDATE runs
            SET run_progress = ?
            WHERE run_id = ?
        """, [float(run_progress), run_id])


def insert_market_round_rows(con, records):
    data = []
    for r in records:
        data.append((
            r["run_id"],
            r["round_number"],
            r["p_t"],
            r["best_bid"],
            r["best_ask"],
            r["volume"],
            r["n_trades"],
            r["demand_at_p"],
            r["supply_at_p"],
            r["n_active_buyers"],
            r["n_active_sellers"],
            r["n_active_total"],
            r["bid_depth_total"],
            r["ask_depth_total"],
            r["price_levels_bid"],
            r["price_levels_ask"]
        ))

    con.executemany("""
        INSERT INTO market_round (
            run_id,
            round_number,
            p_t,
            best_bid,
            best_ask,
            volume,
            n_trades,
            demand_at_p,
            supply_at_p,
            n_active_buyers,
            n_active_sellers,
            n_active_total,
            bid_depth_total,
            ask_depth_total,
            price_levels_bid,
            price_levels_ask
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
