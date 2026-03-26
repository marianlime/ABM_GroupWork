"""Lightweight helpers for running and monitoring sweep comparisons."""

import os
import tempfile
import time
import uuid
from pathlib import Path
import duckdb
import pandas as pd
from Misc.defaults import WEALTH_INFORMED_COLUMN, WEALTH_ZI_COLUMN
from Database.database_creation import create_database


SWEEP_PARAM_SUBPLOTS = [
    ("mean_qty_aggression", "Qty Aggression"),
    ("mean_signal_aggression", "Signal Aggression"),
    ("mean_info_param_parameterised_informed", "Info Param (informed)"),
]
SWEEP_STD_SUBPLOTS = [
    ("std_qty_aggression", "Qty Aggression Std Dev"),
    ("std_signal_aggression", "Signal Aggression Std Dev"),
    ("std_info_param_parameterised_informed", "Info Param Std Dev"),
]


def _connect_read_only_with_retry(
    db_path: str,
    *,
    max_attempts: int = 8,
    delay_seconds: float = 0.1,
):
    """Retry read-only DuckDB opens to avoid transient lock races with sweep writers."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return duckdb.connect(str(db_path), read_only=True)
        except Exception as exc:
            last_exc = exc
            if "being used by another process" not in str(exc).lower() or attempt == max_attempts - 1:
                raise
            time.sleep(delay_seconds)
    raise last_exc


def prepare_sweep_plot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and normalize sweep data to the columns the comparison plots actually consume."""
    if df is None or df.empty:
        return pd.DataFrame()

    prepared_df = df.copy()
    if "generation" not in prepared_df.columns and "generation_id" in prepared_df.columns:
        prepared_df["generation"] = prepared_df["generation_id"].astype(int) - 1

    needed_cols = {"generation", "generation_id", WEALTH_INFORMED_COLUMN, WEALTH_ZI_COLUMN}
    for metric_key, _ in SWEEP_PARAM_SUBPLOTS:
        needed_cols.add(metric_key)
    for metric_key, _ in SWEEP_STD_SUBPLOTS:
        needed_cols.add(metric_key)

    keep_cols = [col for col in prepared_df.columns if col in needed_cols]
    if not keep_cols:
        return pd.DataFrame()

    prepared_df = prepared_df[keep_cols].copy()
    if "generation" in prepared_df.columns:
        prepared_df["generation"] = prepared_df["generation"].astype(int)
    if "generation_id" in prepared_df.columns:
        prepared_df["generation_id"] = prepared_df["generation_id"].astype(int)

    numeric_cols = [col for col in prepared_df.columns if col not in {"generation", "generation_id"}]
    for col in numeric_cols:
        prepared_df[col] = pd.to_numeric(prepared_df[col], errors="coerce")

    sort_cols = ["generation"] if "generation" in prepared_df.columns else ["generation_id"]
    return prepared_df.sort_values(sort_cols).reset_index(drop=True)


def make_temp_duckdb_path(prefix: str) -> str:
    """Return a unique DuckDB temp path without pre-creating an empty file."""
    temp_dir = Path(tempfile.gettempdir())
    return str(temp_dir / f"{prefix}_{uuid.uuid4().hex}.duckdb")


def run_single_sweep_process(args: tuple) -> tuple:
    """Run one sweep experiment in its own process and isolated temporary DuckDB."""
    run_index, sweep_name, sweep_title, settings, label, overrides, tmp_db = args
    graphs_only = bool(settings.get("graphs_only", False))
    try:
        run_overrides = dict(overrides)
        run_overrides["experiment_name"] = label
        run_overrides["experiment_type"] = f"Sweep: {sweep_name}"
        run_overrides["run_notes"] = sweep_title

        if not graphs_only:
            try:
                os.unlink(tmp_db)
            except OSError:
                pass
            run_overrides["db_path"] = tmp_db

        from main import run_experiment

        result = run_experiment(
            config_overrides=run_overrides,
            progress_callback=None,
            run_analysis=False,
            disable_db_writes=graphs_only,
        )
        run_df = prepare_sweep_plot_dataframe(
            result["generation_counts_df"].reset_index(drop=True).copy()
        )
        return run_index, {
            "label": label,
            "experiment_id": result.get("experiment_id"),
            "data": run_df,
            "temp_db_path": tmp_db if not graphs_only else None,
        }
    except Exception:
        if tmp_db:
            try:
                os.unlink(tmp_db)
            except OSError:
                pass
            try:
                os.unlink(f"{tmp_db}.wal")
            except OSError:
                pass
        raise


def peek_partial_sweep_progress(
    temp_db_path: str,
    experiment_id: str | None = None,
) -> tuple[str | None, int]:
    """Return the experiment id and latest completed generation in a temp sweep DB."""
    db_file = Path(temp_db_path)
    if not db_file.exists():
        return None, 0

    con = None
    try:
        con = _connect_read_only_with_retry(db_file)
        if experiment_id is None:
            row = con.execute(
                """
                SELECT experiment_id, MAX(generation_id) AS max_generation_id
                FROM generations
                GROUP BY experiment_id
                ORDER BY max_generation_id DESC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None, 0
            return str(row[0]), int(row[1] or 0)

        row = con.execute(
            """
            SELECT MAX(generation_id)
            FROM generations
            WHERE experiment_id = ?
            """,
            [experiment_id],
        ).fetchone()
        return experiment_id, int(row[0] or 0) if row else 0
    except Exception:
        return None, 0
    finally:
        if con is not None:
            con.close()


def load_partial_sweep_run(
    temp_db_path: str,
    experiment_id: str | None = None,
) -> tuple[str | None, pd.DataFrame]:
    """Load the currently available generation comparison data for one temporary sweep DB."""
    db_file = Path(temp_db_path)
    if not db_file.exists():
        return None, pd.DataFrame()

    con = None
    try:
        con = _connect_read_only_with_retry(db_file)
        if experiment_id is None:
            experiment_row = con.execute(
                """
                SELECT experiment_id
                FROM experiments
                ORDER BY creation_time DESC
                LIMIT 1
                """
            ).fetchone()
            if not experiment_row:
                return None, pd.DataFrame()
            experiment_id = str(experiment_row[0])

        generations_df = con.execute(
            """
            SELECT
                generation_id,
                mean_qty_aggression,
                mean_signal_aggression
            FROM generations
            WHERE experiment_id = ?
            ORDER BY generation_id
            """,
            [experiment_id],
        ).fetchdf()
        if generations_df.empty:
            return experiment_id, pd.DataFrame()

        wealth_df = con.execute(
            """
            WITH final_round AS (
                SELECT generation_id, MAX(round_number) AS final_round_number
                FROM agent_round
                WHERE experiment_id = ?
                GROUP BY generation_id
            )
            SELECT
                ar.generation_id,
                ap.strategy_type,
                AVG(ar.cash_end + ar.inventory_end * COALESCE(mr.p_t, fs.price, 0)) AS mean_wealth
            FROM final_round fr
            JOIN agent_round ar
              ON fr.generation_id = ar.generation_id
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
            WHERE ar.experiment_id = ?
            GROUP BY ar.generation_id, ap.strategy_type
            ORDER BY ar.generation_id, ap.strategy_type
            """,
            [experiment_id, experiment_id],
        ).fetchdf()
        info_df = con.execute(
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
            [experiment_id],
        ).fetchdf()
        diversity_df = con.execute(
            """
            SELECT
                generation_id,
                STDDEV_SAMP(qty_aggression) AS std_qty_aggression,
                STDDEV_SAMP(signal_aggression) AS std_signal_aggression,
                STDDEV_SAMP(CASE
                    WHEN strategy_type = 'parameterised_informed' THEN info_param
                    ELSE NULL
                END) AS std_info_param_parameterised_informed
            FROM agent_population
            WHERE experiment_id = ?
            GROUP BY generation_id
            ORDER BY generation_id
            """,
            [experiment_id],
        ).fetchdf()

        comparison_df = generations_df.copy()
        if not wealth_df.empty:
            comparison_df = comparison_df.merge(
                wealth_df.pivot(index="generation_id", columns="strategy_type", values="mean_wealth")
                .reset_index()
                .rename(columns={
                    "parameterised_informed": "mean_wealth_parameterised_informed",
                    "zi": "mean_wealth_zi",
                }),
                on="generation_id",
                how="left",
            )
        if not info_df.empty:
            comparison_df = comparison_df.merge(
                info_df.pivot(index="generation_id", columns="strategy_type", values="mean_info_param")
                .reset_index()
                .rename(columns={
                    "parameterised_informed": "mean_info_param_parameterised_informed",
                    "zi": "mean_info_param_zi",
                }),
                on="generation_id",
                how="left",
            )
        if not diversity_df.empty:
            comparison_df = comparison_df.merge(diversity_df, on="generation_id", how="left")
        comparison_df["experiment_id"] = experiment_id
        return experiment_id, prepare_sweep_plot_dataframe(comparison_df)
    except Exception:
        return None, pd.DataFrame()
    finally:
        if con is not None:
            con.close()


def merge_experiments_from_temp_dbs(source_runs: list[tuple[str, str]], target_db_path: str) -> None:
    """Copy completed temp-db experiments into the main GUI database using one target connection."""
    if not source_runs:
        return

    create_database(target_db_path)
    con = duckdb.connect(target_db_path)
    try:
        for attach_index, (source_db_path, experiment_id) in enumerate(source_runs):
            if not source_db_path or not experiment_id:
                continue

            source_literal = source_db_path.replace("'", "''")
            alias = f"sweep_src_{attach_index}"
            con.execute(f"ATTACH '{source_literal}' AS {alias}")
            try:
                discovered_table_rows = con.execute(
                    f"""
                    SELECT table_name
                    FROM duckdb_tables()
                    WHERE database_name = '{alias}' AND schema_name = 'main'
                    ORDER BY table_name
                    """
                ).fetchall()
                discovered_tables = [str(row[0]) for row in discovered_table_rows]
                preferred_order = [
                    "experiments",
                    "generations",
                    "gbm_config",
                    "fundamental_series",
                    "market_round",
                    "agent_population",
                    "agent_round",
                    "trade_execution",
                ]
                ordered_tables = [name for name in preferred_order if name in discovered_tables]
                ordered_tables.extend(
                    name for name in discovered_tables if name not in ordered_tables
                )
                for table_name in ordered_tables:
                    column_rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                    column_names = [str(row[1]) for row in column_rows]
                    if "experiment_id" not in column_names:
                        continue
                    column_sql = ", ".join(f'"{name}"' for name in column_names)
                    con.execute(
                        f"""
                        INSERT INTO "{table_name}" ({column_sql})
                        SELECT {column_sql}
                        FROM {alias}."{table_name}"
                        WHERE experiment_id = ?
                        """,
                        [experiment_id],
                    )
                con.execute(f"DETACH {alias}")
            except Exception:
                con.execute(f"DETACH {alias}")
                raise
    finally:
        con.close()
