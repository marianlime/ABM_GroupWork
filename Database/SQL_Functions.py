"""
DuckDB persistence layer: individual insert/update helpers for every table in
the schema, plus AsyncDuckDBWriter for non-blocking background writes.
"""

import json
import queue
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

from Database.database_creation import create_database


SQL_TAB_PREVIEW_ROWS = 1000


def _json_default(value):
    """JSON serialiser fallback that converts sets to sorted lists."""
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _connect_with_retry(db_path: str, *, max_attempts: int = 40, delay_seconds: float = 0.1):
    """Retry opening a DuckDB connection for short-lived Windows file-lock races."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return duckdb.connect(db_path)
        except Exception as exc:
            last_exc = exc
            if "being used by another process" not in str(exc).lower() or attempt == max_attempts - 1:
                raise
            time.sleep(delay_seconds)
    raise last_exc


def ensure_database_exists(db_path: str) -> bool:
    """Create the DuckDB file and schema if needed, returning whether it was newly created."""
    db_file = Path(db_path)
    database_created = not db_file.exists()
    create_database(str(db_file))
    return database_created


def humanise_sql_object_name(name: str) -> str:
    """Convert snake_case SQL object names into a friendlier title for the GUI."""
    replacements = {
        "id": "ULID",
        "dt": "Date",
        "ts": "Timestamp",
        "num": "Number",
        "qty": "Quantity",
        "gbm": "GBM",
        "sql": "SQL",
        "avg": "Average",
    }
    parts = []
    for part in name.split("_"):
        parts.append(replacements.get(part, part.capitalize()))
    return " ".join(parts)


def _fetch_sql_objects(
    con,
    *,
    selected_experiment_id: str | None = None,
    selected_generation_id: int | None = None,
    experiment_ids_sql: str | None = None,
) -> dict:
    """Load SQL table/view previews for the SQL tab with optional experiment/generation filtering."""
    sql_objects = {}
    object_rows = con.execute(
        """
        SELECT
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY
            CASE WHEN table_type = 'BASE TABLE' THEN 0 ELSE 1 END,
            table_name
        """
    ).fetchall()

    for object_name, object_type in object_rows:
        try:
            column_rows = con.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = ?
                ORDER BY ordinal_position
                """,
                [object_name],
            ).fetchall()
            ordered_column_names = [str(row[0]) for row in column_rows]
            column_name_set = set(ordered_column_names)
            where_clauses = []
            query_params = []

            if experiment_ids_sql is not None and "experiment_id" in column_name_set:
                where_clauses.append(f"experiment_id IN ({experiment_ids_sql})")
            elif selected_experiment_id is not None and "experiment_id" in column_name_set:
                where_clauses.append("experiment_id = ?")
                query_params.append(selected_experiment_id)

            if selected_generation_id is not None and "generation_id" in column_name_set:
                where_clauses.append("generation_id = ?")
                query_params.append(selected_generation_id)

            base_query = f'SELECT * FROM "{object_name}"'
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)

            order_columns = [
                column_name
                for column_name in ["experiment_id", "generation_id", "round_number", "agent_id", "trade_id"]
                if column_name in column_name_set
            ]
            if order_columns:
                base_query += " ORDER BY " + ", ".join(order_columns)

            total_rows = int(
                con.execute(
                    f"SELECT COUNT(*) FROM ({base_query}) AS filtered_object",
                    query_params,
                ).fetchone()[0]
            )
            sql_objects[str(object_name)] = {
                "display_name": humanise_sql_object_name(str(object_name)),
                "type": str(object_type),
                "row_count": total_rows,
                "preview_rows": min(total_rows, SQL_TAB_PREVIEW_ROWS),
                "data": con.execute(
                    f"{base_query} LIMIT {SQL_TAB_PREVIEW_ROWS}",
                    query_params,
                ).fetchdf(),
            }
        except Exception as exc:
            sql_objects[str(object_name)] = {
                "display_name": humanise_sql_object_name(str(object_name)),
                "type": str(object_type),
                "row_count": 0,
                "preview_rows": 0,
                "data": pd.DataFrame({"error": [str(exc)]}),
            }

    return sql_objects


def load_database_payload(db_path: str, experiment_id=None, generation_id=None) -> dict:
    """Load the GUI payload for a single selected experiment and generation."""
    db_file = Path(db_path)
    database_created = ensure_database_exists(str(db_file))

    con = duckdb.connect(str(db_file))
    try:
        experiments_df = con.execute(
            """
            SELECT experiment_id, experiment_name, experiment_type, creation_time, completion_time, n_generations, n_rounds, n_agents, algorithm_name
            FROM experiments
            ORDER BY creation_time DESC
            """
        ).fetchdf()

        selected_experiment_id = experiment_id
        if experiments_df.empty:
            selected_experiment_id = None
        elif selected_experiment_id not in set(experiments_df["experiment_id"].tolist()):
            selected_experiment_id = str(experiments_df.iloc[0]["experiment_id"])

        generations_df = pd.DataFrame()
        wealth_history_df = pd.DataFrame()
        mean_info_param_df = pd.DataFrame()
        strategy_generation_df = pd.DataFrame()
        market_history_df = pd.DataFrame()
        market_summary_df = pd.DataFrame()
        strategy_profit_round_df = pd.DataFrame()
        volume_share_round_df = pd.DataFrame()
        agent_strategy_evolution_df = pd.DataFrame()
        agent_profit_loss_df = pd.DataFrame()
        agent_fill_rate_df = pd.DataFrame()
        agent_inventory_risk_df = pd.DataFrame()
        agent_inventory_turnover_df = pd.DataFrame()
        agent_relative_performance_df = pd.DataFrame()
        agent_signal_accuracy_df = pd.DataFrame()
        agent_volume_share_df = pd.DataFrame()
        agent_aggressiveness_spread_df = pd.DataFrame()
        agent_order_count_df = pd.DataFrame()
        agent_behavior_change_df = pd.DataFrame()
        agent_execution_price_deviation_df = pd.DataFrame()
        agent_avg_trade_size_df = pd.DataFrame()
        agent_info_param_history_df = pd.DataFrame()
        agent_strategy_param_history_df = pd.DataFrame()
        population_df = pd.DataFrame()
        agent_round_df = pd.DataFrame()
        trade_execution_df = pd.DataFrame()
        selected_generation_id = generation_id

        if selected_experiment_id is not None:
            generations_df = con.execute(
                """
                SELECT
                    generation_id,
                    generation_status,
                    generation_progress,
                    creation_time,
                    completion_time,
                    mean_qty_aggression,
                    mean_signal_aggression
                FROM generations
                WHERE experiment_id = ?
                ORDER BY generation_id
                """,
                [selected_experiment_id],
            ).fetchdf()

            if generations_df.empty:
                selected_generation_id = None
            elif selected_generation_id not in set(generations_df["generation_id"].tolist()):
                selected_generation_id = int(generations_df.iloc[-1]["generation_id"])

            wealth_history_df = con.execute(
                """
                WITH final_round AS (
                    SELECT
                        experiment_id,
                        generation_id,
                        MAX(round_number) AS final_round_number
                    FROM agent_round
                    WHERE experiment_id = ?
                    GROUP BY experiment_id, generation_id
                )
                SELECT
                    ar.generation_id,
                    ap.strategy_type,
                    AVG(ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) AS mean_wealth
                FROM final_round fr
                JOIN agent_round ar
                  ON fr.experiment_id = ar.experiment_id
                 AND fr.generation_id = ar.generation_id
                 AND fr.final_round_number = ar.round_number
                JOIN agent_population ap
                  ON ar.experiment_id = ap.experiment_id
                 AND ar.generation_id = ap.generation_id
                 AND ar.agent_id = ap.agent_id
                LEFT JOIN market_round mr
                  ON ar.experiment_id = mr.experiment_id
                 AND ar.generation_id = mr.generation_id
                 AND ar.round_number = mr.round_number
                LEFT JOIN fundamental_series fs
                  ON ar.experiment_id = fs.experiment_id
                 AND ar.generation_id = fs.generation_id
                 AND ar.round_number = fs.round_number
                GROUP BY ar.generation_id, ap.strategy_type
                ORDER BY ar.generation_id, ap.strategy_type
                """,
                [selected_experiment_id],
            ).fetchdf()

            mean_info_param_df = con.execute(
                """
                SELECT
                    generation_id,
                    strategy_type,
                    AVG(info_param) AS mean_info_param
                FROM agent_population
                WHERE experiment_id = ?
                GROUP BY generation_id, strategy_type
                ORDER BY generation_id, strategy_type
                """,
                [selected_experiment_id],
            ).fetchdf()

            strategy_generation_df = con.execute(
                """
                SELECT
                    generation_id,
                    strategy_type,
                    total_agent_rounds,
                    avg_profit_loss_per_gen,
                    avg_fill_rate_per_gen,
                    avg_aggressiveness_per_gen,
                    avg_signal_accuracy_per_gen,
                    avg_inventory_turnover_per_gen,
                    avg_execution_price_deviation_per_gen,
                    avg_volume_share_per_gen,
                    avg_trade_size_per_gen,
                    avg_inventory_risk_per_gen
                FROM strategy_performance_per_generation
                WHERE experiment_id = ?
                ORDER BY generation_id, strategy_type
                """,
                [selected_experiment_id],
            ).fetchdf()

            agent_strategy_evolution_df = con.execute(
                """
                SELECT
                    generation_id,
                    strategy_type,
                    avg_profit_loss_per_gen,
                    profit_change_from_prev_gen
                FROM strategy_evolution_across_generations
                WHERE experiment_id = ?
                ORDER BY generation_id, strategy_type
                """,
                [selected_experiment_id],
            ).fetchdf()

            agent_info_param_history_df = con.execute(
                """
                SELECT
                    generation_id,
                    agent_id,
                    strategy_type,
                    info_param
                FROM agent_population
                WHERE experiment_id = ?
                  AND strategy_type = 'parameterised_informed'
                  AND info_param IS NOT NULL
                ORDER BY agent_id, generation_id
                """,
                [selected_experiment_id],
            ).fetchdf()

            agent_strategy_param_history_df = con.execute(
                """
                SELECT
                    generation_id,
                    agent_id,
                    strategy_type,
                    qty_aggression,
                    signal_aggression
                FROM agent_population
                WHERE experiment_id = ?
                  AND strategy_type = 'parameterised_informed'
                ORDER BY agent_id, generation_id
                """,
                [selected_experiment_id],
            ).fetchdf()

        if selected_experiment_id is not None and selected_generation_id is not None:
            market_history_df = con.execute(
                """
                SELECT
                    mr.round_number,
                    mr.p_t,
                    fs.price AS fundamental_price,
                    mr.best_bid,
                    mr.best_ask,
                    mr.volume,
                    mr.n_trades,
                    mr.demand_at_p,
                    mr.supply_at_p,
                    mr.n_active_buyers,
                    mr.n_active_sellers,
                    mr.n_active_total,
                    mr.bid_depth_total,
                    mr.ask_depth_total,
                    mr.price_levels_bid,
                    mr.price_levels_ask
                FROM market_round mr
                LEFT JOIN fundamental_series fs
                  ON mr.experiment_id = fs.experiment_id
                 AND mr.generation_id = fs.generation_id
                 AND mr.round_number = fs.round_number
                WHERE mr.experiment_id = ? AND mr.generation_id = ?
                ORDER BY mr.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            market_summary_df = con.execute(
                """
                SELECT
                    round_number,
                    max_bid,
                    max_sell,
                    min_bid,
                    min_sell,
                    bid_price_q2,
                    ask_price_q3,
                    fundamental_price,
                    mid_price
                FROM market_round_summary
                WHERE experiment_id = ? AND generation_id = ?
                ORDER BY round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            strategy_profit_round_df = con.execute(
                """
                SELECT
                    round_number,
                    strategy_type,
                    avg_wealth,
                    avg_profit_loss,
                    avg_fill_rate,
                    avg_aggressiveness,
                    avg_signal_accuracy,
                    avg_inventory_turnover,
                    avg_execution_price_deviation,
                    avg_volume_share,
                    avg_trade_size,
                    avg_inventory_risk
                FROM strategy_performance_per_round
                WHERE experiment_id = ? AND generation_id = ?
                ORDER BY round_number, strategy_type
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            volume_share_round_df = con.execute(
                """
                SELECT
                    ar.round_number,
                    ap.strategy_type,
                    AVG(CASE WHEN mr.volume > 0 THEN ar.executed_qty / mr.volume ELSE 0 END) AS avg_volume_share
                FROM agent_round ar
                JOIN agent_population ap
                  ON ar.experiment_id = ap.experiment_id
                 AND ar.generation_id = ap.generation_id
                 AND ar.agent_id = ap.agent_id
                JOIN market_round mr
                  ON ar.experiment_id = mr.experiment_id
                 AND ar.generation_id = mr.generation_id
                 AND ar.round_number = mr.round_number
                WHERE ar.experiment_id = ? AND ar.generation_id = ?
                GROUP BY ar.round_number, ap.strategy_type
                ORDER BY ar.round_number, ap.strategy_type
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            population_df = con.execute(
                """
                SELECT
                    agent_id,
                    strategy_type,
                    info_param,
                    qty_aggression,
                    signal_aggression,
                    group_label,
                    initial_cash,
                    initial_shares
                FROM agent_population
                WHERE experiment_id = ? AND generation_id = ?
                ORDER BY agent_id
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_round_df = con.execute(
                """
                SELECT
                    round_number,
                    agent_id,
                    action,
                    signal,
                    signal_error,
                    limit_price,
                    order_qty,
                    aggressiveness,
                    executed_qty,
                    executed_price_avg,
                    fill_ratio,
                    cash_end,
                    inventory_end
                FROM agent_round
                WHERE experiment_id = ? AND generation_id = ?
                ORDER BY round_number, agent_id
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_profit_loss_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.profit_loss
                FROM agent_profit_loss_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_fill_rate_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.fill_rate
                FROM agent_fill_rate_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_inventory_risk_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.inventory_risk
                FROM agent_inventory_risk_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_inventory_turnover_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.inventory_turnover
                FROM agent_inventory_turnover_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_relative_performance_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.relative_profit_loss
                FROM agent_relative_performance_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_signal_accuracy_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.signal_accuracy
                FROM agent_signal_accuracy_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_volume_share_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.volume_share
                FROM agent_volume_share_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_aggressiveness_spread_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.aggressiveness_spread
                FROM agent_aggressiveness_spread_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_order_count_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.order_count
                FROM agent_order_count_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_behavior_change_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.behavior_change
                FROM agent_behavior_change_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_execution_price_deviation_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.execution_price_deviation
                FROM agent_execution_price_deviation_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            agent_avg_trade_size_df = con.execute(
                """
                SELECT
                    v.round_number,
                    v.agent_id,
                    ap.strategy_type,
                    v.avg_trade_size
                FROM agent_avg_trade_size_per_round v
                JOIN agent_population ap
                  ON v.experiment_id = ap.experiment_id
                 AND v.generation_id = ap.generation_id
                 AND v.agent_id = ap.agent_id
                WHERE v.experiment_id = ? AND v.generation_id = ?
                ORDER BY v.agent_id, v.round_number
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

            trade_execution_df = con.execute(
                """
                SELECT
                    round_number,
                    trade_id,
                    buyer_agent_id,
                    seller_agent_id,
                    price,
                    quantity,
                    notional
                FROM trade_execution
                WHERE experiment_id = ? AND generation_id = ?
                ORDER BY round_number, trade_id
                """,
                [selected_experiment_id, selected_generation_id],
            ).fetchdf()

        sql_objects = _fetch_sql_objects(
            con,
            selected_experiment_id=selected_experiment_id,
            selected_generation_id=selected_generation_id,
        )

        return {
            "database_created": database_created,
            "experiments": experiments_df,
            "generations": generations_df,
            "wealth_history": wealth_history_df,
            "mean_info_param": mean_info_param_df,
            "strategy_generation": strategy_generation_df,
            "market_history": market_history_df,
            "market_summary": market_summary_df,
            "strategy_profit_round": strategy_profit_round_df,
            "volume_share_round": volume_share_round_df,
            "agent_strategy_evolution": agent_strategy_evolution_df,
            "agent_profit_loss": agent_profit_loss_df,
            "agent_fill_rate": agent_fill_rate_df,
            "agent_inventory_risk": agent_inventory_risk_df,
            "agent_inventory_turnover": agent_inventory_turnover_df,
            "agent_relative_performance": agent_relative_performance_df,
            "agent_signal_accuracy": agent_signal_accuracy_df,
            "agent_volume_share": agent_volume_share_df,
            "agent_aggressiveness_spread": agent_aggressiveness_spread_df,
            "agent_order_count": agent_order_count_df,
            "agent_behavior_change": agent_behavior_change_df,
            "agent_execution_price_deviation": agent_execution_price_deviation_df,
            "agent_avg_trade_size": agent_avg_trade_size_df,
            "agent_info_param_history": agent_info_param_history_df,
            "agent_strategy_param_history": agent_strategy_param_history_df,
            "sql_objects": sql_objects,
            "population": population_df,
            "agent_round": agent_round_df,
            "trade_execution": trade_execution_df,
            "selected_experiment_id": selected_experiment_id,
            "selected_generation_id": selected_generation_id,
        }
    finally:
        con.close()


def load_comparison_payload(db_path: str, experiment_ids) -> dict:
    """Load the GUI payload used by the multi-experiment comparison view."""
    db_file = Path(db_path)
    ensure_database_exists(str(db_file))
    experiment_ids = [str(experiment_id) for experiment_id in experiment_ids]
    if not experiment_ids:
        return {
            "experiments": pd.DataFrame(),
            "comparison_metrics": pd.DataFrame(),
            "sql_objects": {},
        }

    con = duckdb.connect(str(db_file))
    try:
        experiment_ids_sql = ", ".join(f"'{experiment_id}'" for experiment_id in experiment_ids)

        experiments_df = con.execute(
            f"""
            SELECT
                experiment_id,
                experiment_name,
                experiment_type,
                creation_time,
                n_generations,
                n_rounds,
                n_agents
            FROM experiments
            WHERE experiment_id IN ({experiment_ids_sql})
            ORDER BY creation_time DESC
            """
        ).fetchdf()

        if experiments_df.empty:
            return {
                "experiments": experiments_df,
                "comparison_metrics": pd.DataFrame(),
                "sql_objects": {},
            }

        generations_df = con.execute(
            f"""
            SELECT
                experiment_id,
                generation_id,
                mean_qty_aggression,
                mean_signal_aggression
            FROM generations
            WHERE experiment_id IN ({experiment_ids_sql})
            ORDER BY experiment_id, generation_id
            """
        ).fetchdf()

        wealth_history_df = con.execute(
            f"""
            WITH final_round AS (
                SELECT
                    experiment_id,
                    generation_id,
                    MAX(round_number) AS final_round_number
                FROM agent_round
                WHERE experiment_id IN ({experiment_ids_sql})
                GROUP BY experiment_id, generation_id
            )
            SELECT
                ar.experiment_id,
                ar.generation_id,
                ap.strategy_type,
                AVG(ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) AS mean_wealth
            FROM final_round fr
            JOIN agent_round ar
              ON fr.experiment_id = ar.experiment_id
             AND fr.generation_id = ar.generation_id
             AND fr.final_round_number = ar.round_number
            JOIN agent_population ap
              ON ar.experiment_id = ap.experiment_id
             AND ar.generation_id = ap.generation_id
             AND ar.agent_id = ap.agent_id
            LEFT JOIN market_round mr
              ON ar.experiment_id = mr.experiment_id
             AND ar.generation_id = mr.generation_id
             AND ar.round_number = mr.round_number
            LEFT JOIN fundamental_series fs
              ON ar.experiment_id = fs.experiment_id
             AND ar.generation_id = fs.generation_id
             AND ar.round_number = fs.round_number
            GROUP BY ar.experiment_id, ar.generation_id, ap.strategy_type
            ORDER BY ar.experiment_id, ar.generation_id, ap.strategy_type
            """
        ).fetchdf()

        mean_info_param_df = con.execute(
            f"""
            SELECT
                experiment_id,
                generation_id,
                strategy_type,
                AVG(info_param) AS mean_info_param
            FROM agent_population
            WHERE experiment_id IN ({experiment_ids_sql})
            GROUP BY experiment_id, generation_id, strategy_type
            ORDER BY experiment_id, generation_id, strategy_type
            """
        ).fetchdf()

        diversity_df = con.execute(
            f"""
            SELECT
                experiment_id,
                generation_id,
                STDDEV_SAMP(qty_aggression) AS std_qty_aggression,
                STDDEV_SAMP(signal_aggression) AS std_signal_aggression,
                STDDEV_SAMP(CASE
                    WHEN strategy_type = 'parameterised_informed' THEN info_param
                    ELSE NULL
                END) AS std_info_param_parameterised_informed
            FROM agent_population
            WHERE experiment_id IN ({experiment_ids_sql})
            GROUP BY experiment_id, generation_id
            ORDER BY experiment_id, generation_id
            """
        ).fetchdf()

        comparison_df = generations_df.copy()
        if not wealth_history_df.empty:
            wealth_pivot = (
                wealth_history_df.pivot(
                    index=["experiment_id", "generation_id"],
                    columns="strategy_type",
                    values="mean_wealth",
                )
                .reset_index()
                .rename(
                    columns={
                        "parameterised_informed": "mean_wealth_parameterised_informed",
                        "zi": "mean_wealth_zi",
                    }
                )
            )
            comparison_df = comparison_df.merge(
                wealth_pivot,
                on=["experiment_id", "generation_id"],
                how="left",
            )

        if not mean_info_param_df.empty:
            info_pivot = (
                mean_info_param_df.pivot(
                    index=["experiment_id", "generation_id"],
                    columns="strategy_type",
                    values="mean_info_param",
                )
                .reset_index()
                .rename(
                    columns={
                        "parameterised_informed": "mean_info_param_parameterised_informed",
                        "zi": "mean_info_param_zi",
                    }
                )
            )
            comparison_df = comparison_df.merge(
                info_pivot,
                on=["experiment_id", "generation_id"],
                how="left",
            )

        if not diversity_df.empty:
            comparison_df = comparison_df.merge(
                diversity_df,
                on=["experiment_id", "generation_id"],
                how="left",
            )

        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(
                ["experiment_id", "generation_id"]
            ).reset_index(drop=True)

        sql_objects = _fetch_sql_objects(
            con,
            experiment_ids_sql=experiment_ids_sql,
        )

        return {
            "experiments": experiments_df,
            "comparison_metrics": comparison_df,
            "sql_objects": sql_objects,
        }
    finally:
        con.close()


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


def insert_trade_execution_rows(con, records):
    if not records:
        return

    data = []
    for r in records:
        data.append((
            r["experiment_id"],
            r["generation_id"],
            r["round_number"],
            r["trade_id"],
            r["buyer_agent_id"],
            r["seller_agent_id"],
            r["price"],
            r["quantity"],
            r["notional"],
        ))

    con.executemany("""
        INSERT INTO trade_execution (
            experiment_id,
            generation_id,
            round_number,
            trade_id,
            buyer_agent_id,
            seller_agent_id,
            price,
            quantity,
            notional
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                              trade_execution_records,
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
        insert_trade_execution_rows(con, trade_execution_records)
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
        con = None
        try:
            con = _connect_with_retry(self.db_path)
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
            if con is not None:
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
