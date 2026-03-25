"""
DuckDB persistence layer: individual insert/update helpers for every table in
the schema, plus AsyncDuckDBWriter for non-blocking background writes.
"""

import json
import queue
import threading

import duckdb


def _json_default(value):
    """JSON serialiser fallback that converts sets to sorted lists."""
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def insert_experiment_row(con,
                          experiment_id,
                          experiment_name,
                          experiment_type,
                          creation_time,
                          completion_time,
                          run_notes,
                          n_generations,
                          n_rounds,
                          fundamental_source,
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
                          info_param_distribution_type,
                          distribution_data,
                          signal_generator_noise_distribution,
                          algorithm_name,
                          algorithm_params,
                          bias):
    """Insert a single row into the experiments table, skipping columns absent from the schema."""
    experiment_columns = {
        row[1] for row in con.execute("PRAGMA table_info('experiments')").fetchall()
    }

    column_names = [
        "experiment_id",
        "experiment_name",
        "experiment_type",
        "creation_time",
        "completion_time",
        "run_notes",
        "n_generations",
        "n_rounds",
        "fundamental_source",
        "py_vers",
        "code_vers",
        "n_agents",
        "total_starting_cash",
        "total_starting_shares",
        "market_mechanism",
        "pricing_rule",
        "rationing_rule",
        "tie_break_rule",
        "transaction_cost_rate",
    ]
    values = [
        experiment_id,
        experiment_name,
        experiment_type,
        creation_time,
        completion_time,
        run_notes,
        n_generations,
        n_rounds,
        fundamental_source,
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
    ]

    if "noise_parameter_distribution_type" in experiment_columns:
        column_names.append("noise_parameter_distribution_type")
        values.append(info_param_distribution_type)
    if "info_param_distribution_type" in experiment_columns:
        column_names.append("info_param_distribution_type")
        values.append(info_param_distribution_type)

    column_names.extend([
        "distribution_data",
        "signal_generator_noise_distribution",
        "algorithm_name",
        "algorithm_params",
        "bias",
    ])
    values.extend([
        json.dumps(distribution_data, default=_json_default),
        signal_generator_noise_distribution,
        algorithm_name,
        json.dumps(algorithm_params, default=_json_default),
        bias,
    ])

    placeholders = ", ".join(["?"] * len(column_names))
    con.execute(
        f"INSERT INTO experiments ({', '.join(column_names)}) VALUES ({placeholders})",
        values,
    )


def insert_generation_row(con,
                          experiment_id,
                          generation_id,
                          creation_time,
                          completion_time,
                          generation_status,
                          generation_progress):
    """Insert a new row into the generations table with initial status and progress."""
    con.execute("""
        INSERT INTO generations (
            experiment_id,
            generation_id,
            creation_time,
            completion_time,
            generation_status,
            generation_progress
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        experiment_id,
        int(generation_id),
        creation_time,
        completion_time,
        generation_status,
        generation_progress,
    ])


def update_generation_param_means(con,
                                  experiment_id,
                                  generation_id,
                                  mean_qty_aggression,
                                  mean_signal_aggression):
    """Update the mean strategy-parameter columns on an existing generations row."""
    con.execute("""
        UPDATE generations
        SET mean_qty_aggression = ?,
            mean_signal_aggression = ?
        WHERE experiment_id = ? AND generation_id = ?
    """, [
        float(mean_qty_aggression) if mean_qty_aggression is not None else None,
        float(mean_signal_aggression) if mean_signal_aggression is not None else None,
        experiment_id,
        int(generation_id),
    ])


def insert_gbm_config_row(con, experiment_id, generation_id, S0, volatility, drift, seed):
    """Insert a GBM configuration record for a given experiment and generation."""
    con.execute(
        """INSERT INTO gbm_config VALUES (?, ?, ?, ?, ?, ?)""",
        [experiment_id, int(generation_id), S0, drift, volatility, seed]
    )


def insert_fundamental_series(con, experiment_id, generation_id, fundamental_series):
    """Bulk-insert the fundamental price path for a generation into fundamental_series."""
    data = []
    for r, p in fundamental_series:
        if isinstance(p, tuple):
            raise TypeError(f"Expected numeric price, got tuple at round {r}: {p}")
        data.append((experiment_id, int(generation_id), int(r), float(p)))

    con.executemany(
        """
        INSERT INTO fundamental_series (experiment_id, generation_id, round_number, price)
        VALUES (?, ?, ?, ?)
        """,
        data
    )


def insert_agent_population(con, experiment_id, generation_id, agents):
    """Bulk-insert all agent endowment and parameter records for a generation."""
    data = []
    for agent_id, agent in agents.items():
        group_label = "noise" if agent.trader_type == "zi" else "informed"
        params = getattr(agent, "strategy_params", {}) or {}

        qty_aggression = params.get("qty_aggression", None)
        signal_aggression = params.get("signal_aggression", None)

        data.append((
            experiment_id,
            int(generation_id),
            int(agent_id),
            str(agent.trader_type),
            float(agent.info_param),
            float(qty_aggression) if qty_aggression is not None else None,
            float(signal_aggression) if signal_aggression is not None else None,
            group_label,
            float(agent.cash),
            float(agent.shares),
        ))

    con.executemany("""
        INSERT INTO agent_population (
            experiment_id,
            generation_id,
            agent_id,
            strategy_type,
            info_param,
            qty_aggression,
            signal_aggression,
            group_label,
            initial_cash,
            initial_shares
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)


def insert_agent_round_rows(con, records):
    """Bulk-insert per-agent, per-round trading records into agent_round."""
    data = []
    for r in records:
        data.append((
            r["experiment_id"],
            r["generation_id"],
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
            experiment_id,
            generation_id,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)


def update_generation_progress(con,
                               experiment_id,
                               generation_id,
                               generation_progress,
                               generation_status=None,
                               completion_time=None):
    """Update progress and optionally status/completion_time on a generations row."""
    if generation_status is not None and completion_time is not None:
        con.execute("""
        UPDATE generations
        SET generation_progress = ?, generation_status = ?, completion_time = ?
        WHERE experiment_id = ? AND generation_id = ?
        """, [float(generation_progress), generation_status, completion_time, experiment_id, int(generation_id)])

    elif generation_status is not None:
        con.execute("""
            UPDATE generations
            SET generation_progress = ?, generation_status = ?
            WHERE experiment_id = ? AND generation_id = ?
        """, [float(generation_progress), generation_status, experiment_id, int(generation_id)])

    else:
        con.execute("""
            UPDATE generations
            SET generation_progress = ?
            WHERE experiment_id = ? AND generation_id = ?
        """, [float(generation_progress), experiment_id, int(generation_id)])


def insert_market_round_rows(con, records):
    """Bulk-insert per-round market clearing records into market_round."""
    data = []
    for r in records:
        data.append((
            r["experiment_id"],
            r["generation_id"],
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
            r["price_levels_ask"],
        ))

    con.executemany("""
        INSERT INTO market_round (
            experiment_id,
            generation_id,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)


def persist_generation_bundle(con,
                              experiment_id,
                              generation_id,
                              S0,
                              volatility,
                              drift,
                              seed,
                              fundamental_series,
                              agents,
                              market_round_records,
                              agent_round_records,
                              completion_time,
                              mean_qty_aggression,
                              mean_signal_aggression):
    """Atomically write all generation data (GBM config, fundamental path, agents, market and agent rounds) in a single transaction."""
    con.execute("BEGIN TRANSACTION")
    try:
        insert_gbm_config_row(con, experiment_id, generation_id, S0, volatility, drift, seed)
        insert_fundamental_series(con, experiment_id, generation_id, fundamental_series)
        insert_agent_population(con, experiment_id, generation_id, agents)
        insert_market_round_rows(con, market_round_records)
        insert_agent_round_rows(con, agent_round_records)
        update_generation_progress(
            con,
            experiment_id,
            generation_id,
            100.0,
            generation_status="COMPLETED",
            completion_time=completion_time,
        )
        update_generation_param_means(
            con,
            experiment_id,
            generation_id,
            mean_qty_aggression,
            mean_signal_aggression,
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


class AsyncDuckDBWriter:
    """Serialises database writes to a background thread so the simulation loop is never blocked."""

    def __init__(self, db_path: str, max_queue_size: int = 2):
        self.db_path = db_path
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_token = object()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="duckdb-writer", daemon=True)
        self._thread.start()

    def submit(self, operation: str, *args, **kwargs) -> None:
        while True:
            self._raise_if_error()
            try:
                self._queue.put((operation, args, kwargs), timeout=0.1)
                self._raise_if_error()
                return
            except queue.Full:
                continue

    def close(self) -> None:
        while True:
            self._raise_if_error()
            try:
                self._queue.put((self._stop_token, (), {}), timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._raise_if_error()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Async DuckDB writer failed: {self._error}") from self._error

    def _run(self) -> None:
        con = duckdb.connect(self.db_path)
        try:
            while True:
                operation, args, kwargs = self._queue.get()
                try:
                    if operation is self._stop_token:
                        return
                    self._dispatch(con, operation, args, kwargs)
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._error = exc
        finally:
            con.close()

    def _dispatch(self, con, operation: str, args: tuple, kwargs: dict) -> None:
        if operation == "insert_experiment_row":
            insert_experiment_row(con, *args, **kwargs)
            return
        if operation == "insert_generation_row":
            insert_generation_row(con, *args, **kwargs)
            return
        if operation == "persist_generation_bundle":
            persist_generation_bundle(con, *args, **kwargs)
            return
        if operation == "update_generation_progress":
            update_generation_progress(con, *args, **kwargs)
            return
        raise ValueError(f"Unknown async DB operation: {operation}")
