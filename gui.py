import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QThread, Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from database_creation import create_database
from main import DEFAULT_EXPERIMENT_CONFIG, run_experiment


DEFAULT_DB_PATH = "experiment_results.duckdb"
SQL_TAB_PREVIEW_ROWS = 1000
GRAPH_BACKGROUND = "#2b2b2b"
GRAPH_FOREGROUND = "#ffffff"
STRATEGY_COLORS = {
    "zi": (255, 140, 0),
    "parameterised_informed": (30, 144, 255),
}
AGENT_PERFORMANCE_COLORS = {
    "zi": (255, 140, 0),
    "parameterised_informed": (30, 144, 255),
}


def _humanize_sql_object_name(name: str) -> str:
    replacements = {
        "id": "ID",
        "gbm": "GBM",
        "zi": "ZI",
        "qty": "Qty",
        "sql": "SQL",
        "avg": "Avg",
    }
    return " ".join(replacements.get(part, part.capitalize()) for part in name.split("_"))


def _style_plot(plot):
    plot.showGrid(x=True, y=True, alpha=0.25)


def _set_plot_bottom_label(plot, x_label: str, legend_items=None):
    label_html = f"<span style='color: {GRAPH_FOREGROUND};'>{x_label}</span>"
    if legend_items:
        legend_html = "   ".join(
            f"<span style='color: rgb({color[0]}, {color[1]}, {color[2]});'>&#9632; {name}</span>"
            for name, color in legend_items
        )
        label_html += f"<br><span style='font-size: 10pt;'>{legend_html}</span>"
    plot.setLabel("bottom", label_html)


class DatabaseLoaderWorker(QThread):
    loaded = Signal(dict)
    error = Signal(str)

    def __init__(self, db_path, experiment_id=None, generation_id=None):
        super().__init__()
        self.db_path = db_path
        self.experiment_id = experiment_id
        self.generation_id = generation_id

    def run(self):
        try:
            self.loaded.emit(self._load_payload())
        except Exception as exc:
            self.error.emit(str(exc))

    def _load_payload(self):
        db_file = Path(self.db_path)
        if not db_file.exists():
            raise FileNotFoundError(f"Database file not found: {db_file}")

        create_database(str(db_file))

        con = duckdb.connect(str(db_file))
        try:
            experiments_df = con.execute(
                """
                SELECT
                    experiment_id,
                    experiment_name,
                    experiment_type,
                    creation_time,
                    completion_time,
                    n_generations,
                    n_rounds,
                    n_agents,
                    algorithm_name
                FROM experiments
                ORDER BY creation_time DESC
                """
            ).fetchdf()

            selected_experiment_id = self.experiment_id
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
            sql_objects = {}
            population_df = pd.DataFrame()
            agent_round_df = pd.DataFrame()
            selected_generation_id = self.generation_id

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
                        mean_signal_aggression,
                        mean_threshold,
                        mean_signal_clip
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
                        avg_volume_share_per_gen
                        ,
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
                        signal_aggression,
                        threshold,
                        signal_clip
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
                        mr.n_trades
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
                        threshold,
                        signal_clip,
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
                        v.aggressiveness,
                        v.market_spread
                    FROM agent_aggressiveness_vs_spread v
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
                        SUM(v.order_count) AS order_count
                    FROM agent_order_type_distribution_per_round v
                    JOIN agent_population ap
                      ON v.experiment_id = ap.experiment_id
                     AND v.generation_id = ap.generation_id
                     AND v.agent_id = ap.agent_id
                    WHERE v.experiment_id = ? AND v.generation_id = ?
                    GROUP BY v.round_number, v.agent_id, ap.strategy_type
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
                        v.aggressiveness_change,
                        v.order_qty_change,
                        v.inventory_change
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
                    column_names = {str(row[0]) for row in column_rows}
                    where_clauses = []
                    query_params = []
                    if selected_experiment_id is not None and "experiment_id" in column_names:
                        where_clauses.append("experiment_id = ?")
                        query_params.append(selected_experiment_id)
                    if selected_generation_id is not None and "generation_id" in column_names:
                        where_clauses.append("generation_id = ?")
                        query_params.append(selected_generation_id)

                    base_query = f'SELECT * FROM "{object_name}"'
                    if where_clauses:
                        base_query += " WHERE " + " AND ".join(where_clauses)

                    total_rows = int(
                        con.execute(
                            f"SELECT COUNT(*) FROM ({base_query}) AS filtered_object",
                            query_params,
                        ).fetchone()[0]
                    )
                    sql_objects[str(object_name)] = {
                        "display_name": _humanize_sql_object_name(str(object_name)),
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
                        "display_name": _humanize_sql_object_name(str(object_name)),
                        "type": str(object_type),
                        "row_count": 0,
                        "preview_rows": 0,
                        "data": pd.DataFrame({"error": [str(exc)]}),
                    }

            return {
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
                "selected_experiment_id": selected_experiment_id,
                "selected_generation_id": selected_generation_id,
            }
        finally:
            con.close()


class ExperimentRunnerWorker(QThread):
    progress = Signal(dict)
    completed = Signal(dict)
    error = Signal(str)

    def __init__(self, config_overrides):
        super().__init__()
        self.config_overrides = config_overrides

    def run(self):
        try:
            result = run_experiment(
                config_overrides=self.config_overrides,
                progress_callback=self.progress.emit,
                run_analysis=True,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class CommandCenter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent-Based Market Simulation")
        self.resize(1600, 980)

        self._current_experiment_id = None
        self._current_generation_id = None
        self._suppress_selection_signals = False
        self.worker = None
        self.run_worker = None
        self.live_generations_df = pd.DataFrame()
        self.live_strategy_generation_df = pd.DataFrame()
        self.live_strategy_round_df = pd.DataFrame()
        self._pending_generation_id = None
        self._last_payload = None
        self.smoothing_window = 1
        self._pending_smoothing_window = 1
        self._generation_slider_timer = QTimer(self)
        self._generation_slider_timer.setSingleShot(True)
        self._generation_slider_timer.setInterval(1000)
        self._generation_slider_timer.timeout.connect(self._apply_debounced_generation_change)
        self._smoothing_slider_timer = QTimer(self)
        self._smoothing_slider_timer.setSingleShot(True)
        self._smoothing_slider_timer.setInterval(1000)
        self._smoothing_slider_timer.timeout.connect(self._apply_debounced_smoothing_change)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll)

        self._build_left_panel(left_layout)

        self.tabs = QTabWidget()
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.tabs)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_panel)
        main_layout.addWidget(right_scroll, stretch=1)

        self.dashboard_tab = QWidget()
        self._build_dashboard_tab()
        self.strategy_performance_tab = QWidget()
        self._build_strategy_performance_tab()
        self.agent_performance_tab = QWidget()
        self._build_agent_performance_tab()
        self.sql_tab = QTabWidget()
        self._build_sql_tab()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.strategy_performance_tab, "Strategy Performance")
        self.tabs.addTab(self.agent_performance_tab, "Agent Performance")
        self.tabs.addTab(self.sql_tab, "SQL Data")

        self.combo_experiment.currentIndexChanged.connect(self._on_experiment_changed)
        self.combo_generation.currentIndexChanged.connect(self._on_generation_changed)
        self.generation_slider.valueChanged.connect(self._on_generation_slider_changed)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        self.checkbox_show_parameterised.toggled.connect(self._refresh_plots_only)
        self.checkbox_show_zi.toggled.connect(self._refresh_plots_only)

        self.refresh_data()

    def _build_left_panel(self, left_layout):
        data_group = QGroupBox("Database Controls")
        data_form = QFormLayout()
        self.input_db_path = QLineEdit(DEFAULT_DB_PATH)
        self.combo_experiment = QComboBox()
        self.combo_generation = QComboBox()
        self.generation_slider = QSlider(Qt.Horizontal)
        self.generation_slider.setEnabled(False)
        self.generation_slider.setMinimum(1)
        self.generation_slider.setMaximum(1)
        self.generation_slider.setTickPosition(QSlider.TicksBelow)
        self.generation_slider.setTickInterval(1)
        self.generation_slider_label = QLabel("Generation slider unavailable")
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setMinimum(1)
        self.smoothing_slider.setMaximum(25)
        self.smoothing_slider.setValue(1)
        self.smoothing_slider.setTickPosition(QSlider.TicksBelow)
        self.smoothing_slider.setTickInterval(1)
        self.smoothing_slider_label = QLabel("Graph smoothing: 1 (off)")
        self.checkbox_show_parameterised = QCheckBox("Show Parameterised")
        self.checkbox_show_parameterised.setChecked(True)
        self.checkbox_show_zi = QCheckBox("Show ZI")
        self.checkbox_show_zi.setChecked(True)
        data_form.addRow("DuckDB File:", self.input_db_path)
        data_form.addRow("Experiment:", self.combo_experiment)
        data_form.addRow("Generation:", self.combo_generation)
        data_form.addRow("Generation Scroll:", self.generation_slider)
        data_form.addRow("", self.generation_slider_label)
        data_form.addRow("Graph Smoothing:", self.smoothing_slider)
        data_form.addRow("", self.smoothing_slider_label)
        data_form.addRow("Series Visibility:", self.checkbox_show_parameterised)
        data_form.addRow("", self.checkbox_show_zi)
        data_group.setLayout(data_form)
        left_layout.addWidget(data_group)

        self.btn_refresh = QPushButton("Refresh Database")
        self.btn_refresh.setStyleSheet(
            "background-color: #2e8b57; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_refresh.clicked.connect(self.refresh_data)
        left_layout.addWidget(self.btn_refresh)

        info_group = QGroupBox("Selection Summary")
        info_layout = QVBoxLayout()
        self.summary_label = QLabel("No experiment loaded.")
        self.summary_label.setWordWrap(True)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        info_layout.addWidget(self.summary_label)
        info_layout.addWidget(self.status_label)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        run_group = QGroupBox("Create New Experiment")
        run_form = QFormLayout()
        self.run_input_name = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["experiment_name"]))
        self.run_input_type = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["experiment_type"]))
        self.run_input_notes = QPlainTextEdit(str(DEFAULT_EXPERIMENT_CONFIG["run_notes"]))
        self.run_input_seed = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["experiment_seed"]))
        self.run_input_generations = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["n_generations"]))
        self.run_input_rounds = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["n_rounds"]))
        self.run_input_zi = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["n_zi_agents"]))
        self.run_input_informed = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["n_parameterised_agents"]))
        self.run_input_cash = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["total_initial_cash"]))
        self.run_input_shares = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["total_initial_shares"]))
        self.run_input_s0 = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["GBM_S0"]))
        self.run_input_volatility = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["GBM_volatility"]))
        self.run_input_drift = QLineEdit(str(DEFAULT_EXPERIMENT_CONFIG["GBM_drift"]))
        self.run_input_mutation = QLineEdit(
            str(DEFAULT_EXPERIMENT_CONFIG["algorithm_params"]["mutation_rate"])
        )
        run_form.addRow("Experiment Name:", self.run_input_name)
        run_form.addRow("Experiment Type:", self.run_input_type)
        run_form.addRow("Run Notes:", self.run_input_notes)
        run_form.addRow("Seed:", self.run_input_seed)
        run_form.addRow("Generations:", self.run_input_generations)
        run_form.addRow("Rounds per Generation:", self.run_input_rounds)
        run_form.addRow("ZI Agents:", self.run_input_zi)
        run_form.addRow("Informed Agents:", self.run_input_informed)
        run_form.addRow("Initial Cash:", self.run_input_cash)
        run_form.addRow("Initial Shares:", self.run_input_shares)
        run_form.addRow("GBM S0:", self.run_input_s0)
        run_form.addRow("GBM Volatility:", self.run_input_volatility)
        run_form.addRow("GBM Drift:", self.run_input_drift)
        run_form.addRow("Mutation Rate:", self.run_input_mutation)
        run_group.setLayout(run_form)
        left_layout.addWidget(run_group)

        self.btn_start_run = QPushButton("Start New Run")
        self.btn_start_run.setStyleSheet(
            "background-color: #1f5fa5; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_start_run.clicked.connect(self.start_new_run)
        left_layout.addWidget(self.btn_start_run)

        self.btn_stop_run = QPushButton("Stop Current Run")
        self.btn_stop_run.setStyleSheet(
            "background-color: #a52a2a; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_stop_run.clicked.connect(self.stop_run)
        self.btn_stop_run.setEnabled(False)
        left_layout.addWidget(self.btn_stop_run)

        self.run_status_label = QLabel("No run started.")
        self.run_status_label.setWordWrap(True)
        left_layout.addWidget(self.run_status_label)
        left_layout.addStretch()

    def _build_dashboard_tab(self):
        pg.setConfigOption("background", GRAPH_BACKGROUND)
        pg.setConfigOption("foreground", GRAPH_FOREGROUND)

        layout = QVBoxLayout(self.dashboard_tab)
        dashboard_scroll = QScrollArea()
        dashboard_scroll.setWidgetResizable(True)
        dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(dashboard_scroll)

        dashboard_content = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_content)
        self.graph_area = pg.GraphicsLayoutWidget()
        self.graph_area.setMinimumHeight(1800)
        dashboard_layout.addWidget(self.graph_area)
        dashboard_scroll.setWidget(dashboard_content)

        self.plot_params_title = "Mean Strategy Parameters Across Generations"
        self.plot_wealth_title = "Mean Wealth by Strategy"
        self.plot_info_param_title = "Mean Info_Param by Strategy"
        self.plot_market_title = "Market Summary by Round"
        self.plot_profit_title = "Average Profit per Round: ZI vs Parameterised"
        self.plot_volume_share_title = "Average Agent Volume Share per Round"

        self.plot_params = self.graph_area.addPlot(title=self._format_plot_title(self.plot_params_title))
        _style_plot(self.plot_params)
        _set_plot_bottom_label(
            self.plot_params,
            "Generation",
            [
                ("Qty Aggression", (255, 140, 0)),
                ("Signal Aggression", (70, 130, 180)),
                ("Threshold", (220, 20, 60)),
                ("Signal Clip", (46, 139, 87)),
            ],
        )
        self.plot_params.setLabel("left", "Mean Parameter Value")
        self.line_qty = self.plot_params.plot(pen=pg.mkPen((255, 140, 0), width=3), name="Qty Aggression")
        self.line_signal = self.plot_params.plot(pen=pg.mkPen((70, 130, 180), width=3), name="Signal Aggression")
        self.line_threshold = self.plot_params.plot(pen=pg.mkPen((220, 20, 60), width=3), name="Threshold")
        self.line_clip = self.plot_params.plot(pen=pg.mkPen((46, 139, 87), width=3), name="Signal Clip")

        self.graph_area.nextRow()

        self.plot_wealth = self.graph_area.addPlot(title=self._format_plot_title(self.plot_wealth_title))
        _style_plot(self.plot_wealth)
        _set_plot_bottom_label(
            self.plot_wealth,
            "Generation",
            [
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
        )
        self.plot_wealth.setLabel("left", "Mean Wealth")
        self.line_informed_wealth = self.plot_wealth.plot(
            pen=pg.mkPen((30, 144, 255), width=3), name="Parameterised Informed"
        )
        self.line_zi_wealth = self.plot_wealth.plot(
            pen=pg.mkPen((205, 92, 92), width=3), name="ZI"
        )

        self.graph_area.nextRow()

        self.plot_info_param = self.graph_area.addPlot(
            title=self._format_plot_title(self.plot_info_param_title)
        )
        _style_plot(self.plot_info_param)
        _set_plot_bottom_label(
            self.plot_info_param,
            "Generation",
            [
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
        )
        self.plot_info_param.setLabel("left", "Mean Info_Param")
        self.line_info_param_informed = self.plot_info_param.plot(
            pen=pg.mkPen((30, 144, 255), width=3), name="Parameterised Informed"
        )
        self.line_info_param_zi = self.plot_info_param.plot(
            pen=pg.mkPen((205, 92, 92), width=3), name="ZI"
        )

        self.graph_area.nextRow()

        self.plot_market = self.graph_area.addPlot(title=self._format_plot_title(self.plot_market_title))
        _style_plot(self.plot_market)
        _set_plot_bottom_label(
            self.plot_market,
            "Round",
            [
                ("Max Bid", (30, 144, 255)),
                ("Max Sell", (220, 20, 60)),
                ("Min Bid", (135, 206, 250)),
                ("Min Sell", (250, 128, 114)),
                ("Bid Price Q2", (65, 105, 225)),
                ("Ask Price Q3", (178, 34, 34)),
                ("Fundamental", (128, 128, 128)),
                ("Mid Price", (255, 255, 255)),
            ],
        )
        self.plot_market.setLabel("left", "Price")
        self.line_max_bid = self.plot_market.plot(pen=pg.mkPen((30, 144, 255), width=2), name="Max Bid")
        self.line_max_sell = self.plot_market.plot(pen=pg.mkPen((220, 20, 60), width=2), name="Max Sell")
        self.line_min_bid = self.plot_market.plot(pen=pg.mkPen((135, 206, 250), width=1), name="Min Bid")
        self.line_min_sell = self.plot_market.plot(pen=pg.mkPen((250, 128, 114), width=1), name="Min Sell")
        self.line_bid_q2 = self.plot_market.plot(
            pen=pg.mkPen((65, 105, 225), width=2, style=Qt.DashLine), name="Bid Price Q2"
        )
        self.line_ask_q3 = self.plot_market.plot(
            pen=pg.mkPen((178, 34, 34), width=2, style=Qt.DashLine), name="Ask Price Q3"
        )
        self.line_fundamental_price = self.plot_market.plot(
            pen=pg.mkPen((128, 128, 128), width=2), name="Fundamental"
        )
        self.line_mid_price = self.plot_market.plot(
            pen=pg.mkPen((0, 0, 0), width=3), name="Mid Price"
        )

        self.graph_area.nextRow()

        self.plot_profit = self.graph_area.addPlot(title=self._format_plot_title(self.plot_profit_title))
        _style_plot(self.plot_profit)
        _set_plot_bottom_label(
            self.plot_profit,
            "Round",
            [
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
        )
        self.plot_profit.setLabel("left", "Average Profit")
        self.line_profit_informed = self.plot_profit.plot(
            pen=pg.mkPen((30, 144, 255), width=3), name="Parameterised Informed"
        )
        self.line_profit_zi = self.plot_profit.plot(
            pen=pg.mkPen((205, 92, 92), width=3), name="ZI"
        )

        self.graph_area.nextRow()

        self.plot_volume_share = self.graph_area.addPlot(title=self._format_plot_title(self.plot_volume_share_title))
        _style_plot(self.plot_volume_share)
        _set_plot_bottom_label(
            self.plot_volume_share,
            "Round",
            [
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
        )
        self.plot_volume_share.setLabel("left", "Average Volume Share")
        self.line_volume_informed = self.plot_volume_share.plot(
            pen=pg.mkPen((30, 144, 255), width=3), name="Parameterised Informed"
        )
        self.line_volume_zi = self.plot_volume_share.plot(
            pen=pg.mkPen((205, 92, 92), width=3), name="ZI"
        )

    def _build_sql_tab(self):
        self.sql_tables = {}
        self.table_experiments = None
        self.table_generations = None
        self.table_population = None
        self.table_market = None
        self.table_agent_round = None
        self.table_strategy = None

    def _build_strategy_performance_tab(self):
        layout = QVBoxLayout(self.strategy_performance_tab)
        performance_scroll = QScrollArea()
        performance_scroll.setWidgetResizable(True)
        performance_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(performance_scroll)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        generation_group = QGroupBox("Strategy Performance Across Generations")
        generation_layout = QVBoxLayout(generation_group)
        self.performance_generation_area = pg.GraphicsLayoutWidget()
        self.performance_generation_area.setMinimumHeight(1500)
        generation_layout.addWidget(self.performance_generation_area)
        content_layout.addWidget(generation_group)

        round_group = QGroupBox("Strategy Performance Across Rounds")
        round_layout = QVBoxLayout(round_group)
        self.performance_round_area = pg.GraphicsLayoutWidget()
        self.performance_round_area.setMinimumHeight(1500)
        round_layout.addWidget(self.performance_round_area)
        content_layout.addWidget(round_group)

        performance_scroll.setWidget(content)

        self.performance_generation_plots = {}
        self.performance_round_plots = {}
        self.performance_generation_curves = {}
        self.performance_round_curves = {}

        generation_metrics = [
            ("avg_wealth_per_gen", "Average Wealth per Generation"),
            ("avg_profit_loss_per_gen", "Average Profit/Loss per Generation"),
            ("avg_fill_rate_per_gen", "Average Fill Rate per Generation"),
            ("avg_aggressiveness_per_gen", "Average Aggressiveness per Generation"),
            ("avg_signal_accuracy_per_gen", "Average Signal Accuracy per Generation"),
            ("avg_inventory_turnover_per_gen", "Average Inventory Turnover per Generation"),
            ("avg_execution_price_deviation_per_gen", "Average Execution Price Deviation per Generation"),
            ("avg_volume_share_per_gen", "Average Volume Share per Generation"),
            ("avg_trade_size_per_gen", "Average Trade Size per Generation"),
            ("avg_inventory_risk_per_gen", "Average Inventory Risk per Generation"),
        ]
        self._initialise_metric_grid(
            graph_area=self.performance_generation_area,
            metric_specs=generation_metrics,
            curve_store=self.performance_generation_curves,
            plot_store=self.performance_generation_plots,
            x_label="Generation",
        )
        self.performance_generation_area.ci.layout.setColumnStretchFactor(0, 1)
        self.performance_generation_area.ci.layout.setColumnStretchFactor(1, 1)

        round_metrics = [
            ("avg_wealth", "Average Wealth per Round"),
            ("avg_profit_loss", "Average Profit/Loss per Round"),
            ("avg_fill_rate", "Average Fill Rate per Round"),
            ("avg_aggressiveness", "Average Aggressiveness per Round"),
            ("avg_signal_accuracy", "Average Signal Accuracy per Round"),
            ("avg_inventory_turnover", "Average Inventory Turnover per Round"),
            ("avg_execution_price_deviation", "Average Execution Price Deviation per Round"),
            ("avg_volume_share", "Average Volume Share per Round"),
            ("avg_trade_size", "Average Trade Size per Round"),
            ("avg_inventory_risk", "Average Inventory Risk per Round"),
        ]
        self._initialise_metric_grid(
            graph_area=self.performance_round_area,
            metric_specs=round_metrics,
            curve_store=self.performance_round_curves,
            plot_store=self.performance_round_plots,
            x_label="Round",
        )
        self.performance_round_area.ci.layout.setColumnStretchFactor(0, 1)
        self.performance_round_area.ci.layout.setColumnStretchFactor(1, 1)

    def _build_agent_performance_tab(self):
        layout = QVBoxLayout(self.agent_performance_tab)
        agent_scroll = QScrollArea()
        agent_scroll.setWidgetResizable(True)
        agent_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(agent_scroll)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        generation_group = QGroupBox("Agent Performance Across Generations")
        generation_layout = QVBoxLayout(generation_group)
        self.agent_generation_area = pg.GraphicsLayoutWidget()
        self.agent_generation_area.setMinimumHeight(900)
        generation_layout.addWidget(self.agent_generation_area)
        content_layout.addWidget(generation_group)

        round_group = QGroupBox("Agent Performance Across Rounds")
        round_layout = QVBoxLayout(round_group)
        self.agent_round_area = pg.GraphicsLayoutWidget()
        self.agent_round_area.setMinimumHeight(2100)
        round_layout.addWidget(self.agent_round_area)
        content_layout.addWidget(round_group)

        agent_scroll.setWidget(content)

        self.agent_generation_plots = {}
        self.agent_generation_curves = {}
        self.agent_round_plots = {}
        self.agent_round_curves = self.agent_round_plots

        profit_change_title = "Profit Change vs Prev Gen"
        profit_change_plot = self.agent_generation_area.addPlot(
            title=self._format_plot_title(profit_change_title)
        )
        _style_plot(profit_change_plot)
        _set_plot_bottom_label(
            profit_change_plot,
            "Generation",
            [
                ("ZI", STRATEGY_COLORS["zi"]),
                ("Parameterised Informed", STRATEGY_COLORS["parameterised_informed"]),
            ],
        )
        profit_change_plot.setLabel("left", "Value")
        self.agent_generation_plots["profit_change_from_prev_gen"] = (
            profit_change_plot,
            profit_change_title,
        )
        self.agent_generation_curves["profit_change_from_prev_gen"] = {
            "zi": profit_change_plot.plot(
                pen=pg.mkPen(STRATEGY_COLORS["zi"], width=3),
                name="ZI",
            ),
            "parameterised_informed": profit_change_plot.plot(
                pen=pg.mkPen(STRATEGY_COLORS["parameterised_informed"], width=3),
                name="Parameterised Informed",
            ),
        }
        generation_line_specs = [
            ("agent_info_param_plot", "Info_Param per Agent", "Info_Param"),
            ("agent_qty_aggression_plot", "Qty_Aggression per Agent", "Qty_Aggression"),
            ("agent_signal_aggression_plot", "Signal_Aggression per Agent", "Signal_Aggression"),
            ("agent_threshold_plot", "Threshold per Agent", "Threshold"),
            ("agent_signal_clip_plot", "Signal_Clip per Agent", "Signal_Clip"),
        ]
        self.agent_info_param_plot = self.agent_generation_area.addPlot(
            title=self._format_plot_title(generation_line_specs[0][1])
        )
        _style_plot(self.agent_info_param_plot)
        self.agent_info_param_plot.setLabel("bottom", "Generation")
        self.agent_info_param_plot.setLabel("left", generation_line_specs[0][2])
        self.agent_generation_area.nextRow()
        self._initialise_agent_generation_line_grid(generation_line_specs[1:])
        self.agent_generation_area.ci.layout.setColumnStretchFactor(0, 1)
        self.agent_generation_area.ci.layout.setColumnStretchFactor(1, 1)

        round_metrics = [
            ("profit_loss", "Profit/Loss per Round"),
            ("fill_rate", "Fill Rate per Round"),
            ("inventory_risk", "Inventory Risk per Round"),
            ("inventory_turnover", "Inventory Turnover per Round"),
            ("relative_profit_loss", "Relative Performance per Round"),
            ("signal_accuracy", "Signal Accuracy per Round"),
            ("volume_share", "Volume Share per Round"),
            ("avg_trade_size", "Avg Trade Size per Round"),
            ("aggressiveness", "Aggressiveness vs Spread"),
            ("market_spread", "Market Spread"),
            ("aggressiveness_change", "Aggressiveness Change"),
            ("order_qty_change", "Order Qty Change"),
            ("inventory_change", "Inventory Change"),
            ("execution_price_deviation", "Execution Price Deviation per Round"),
        ]
        self._initialise_agent_metric_grid(
            metric_specs=round_metrics,
            plot_store=self.agent_round_plots,
            x_label="Round",
        )
        self.agent_round_area.ci.layout.setColumnStretchFactor(0, 1)
        self.agent_round_area.ci.layout.setColumnStretchFactor(1, 1)

    def _initialise_metric_grid(self, graph_area, metric_specs, curve_store, x_label, plot_store=None):
        for idx, (metric_key, title) in enumerate(metric_specs):
            plot = graph_area.addPlot(title=self._format_plot_title(title))
            _style_plot(plot)
            _set_plot_bottom_label(
                plot,
                x_label,
                [
                    ("ZI", STRATEGY_COLORS["zi"]),
                    ("Parameterised Informed", STRATEGY_COLORS["parameterised_informed"]),
                ],
            )
            plot.setLabel("left", "Value")
            if plot_store is not None:
                plot_store[metric_key] = (plot, title)
            curve_store[metric_key] = {
                "zi": plot.plot(
                    pen=pg.mkPen(STRATEGY_COLORS["zi"], width=3),
                    name="ZI",
                ),
                "parameterised_informed": plot.plot(
                    pen=pg.mkPen(STRATEGY_COLORS["parameterised_informed"], width=3),
                    name="Parameterised Informed",
                ),
            }
            if idx % 2 == 1:
                graph_area.nextRow()

    def _initialise_agent_metric_grid(self, metric_specs, plot_store, x_label):
        for idx, (metric_key, title) in enumerate(metric_specs):
            plot = self.agent_round_area.addPlot(
                title=self._format_plot_title(title)
            )
            _style_plot(plot)
            plot.setLabel("bottom", x_label)
            plot.setLabel("left", "Value")
            plot_store[metric_key] = (plot, title)
            if idx % 2 == 1:
                self.agent_round_area.nextRow()

    def _initialise_agent_generation_line_grid(self, plot_specs):
        for idx, (attr_name, title, y_label) in enumerate(plot_specs):
            plot = self.agent_generation_area.addPlot(title=self._format_plot_title(title))
            _style_plot(plot)
            plot.setLabel("bottom", "Generation")
            plot.setLabel("left", y_label)
            setattr(self, attr_name, plot)
            if idx % 2 == 1:
                self.agent_generation_area.nextRow()

    def refresh_data(self):
        db_path = self.input_db_path.text().strip()
        if not db_path:
            QMessageBox.warning(self, "Missing Database", "Please provide a DuckDB file path.")
            return

        self.btn_refresh.setEnabled(False)
        self.status_label.setText("Loading database...")
        self.worker = DatabaseLoaderWorker(
            db_path=db_path,
            experiment_id=self._current_experiment_id,
            generation_id=self._current_generation_id,
        )
        self.worker.loaded.connect(self._apply_payload)
        self.worker.error.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def start_new_run(self):
        if self.run_worker is not None:
            QMessageBox.information(self, "Run In Progress", "A run is already in progress.")
            return
        if self.worker is not None:
            QMessageBox.information(
                self,
                "Database Busy",
                "Please wait for the current database refresh to finish before starting a run.",
            )
            return

        try:
            mutation_rate = float(self.run_input_mutation.text().strip())
            config_overrides = {
                "db_path": self.input_db_path.text().strip(),
                "experiment_name": self.run_input_name.text().strip(),
                "experiment_type": self.run_input_type.text().strip(),
                "run_notes": self.run_input_notes.toPlainText().strip(),
                "experiment_seed": int(self.run_input_seed.text().strip()),
                "n_generations": int(self.run_input_generations.text().strip()),
                "n_rounds": int(self.run_input_rounds.text().strip()),
                "n_zi_agents": int(self.run_input_zi.text().strip()),
                "n_parameterised_agents": int(self.run_input_informed.text().strip()),
                "total_initial_cash": float(self.run_input_cash.text().strip()),
                "total_initial_shares": int(self.run_input_shares.text().strip()),
                "GBM_S0": float(self.run_input_s0.text().strip()),
                "GBM_volatility": float(self.run_input_volatility.text().strip()),
                "GBM_drift": float(self.run_input_drift.text().strip()),
            }
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Input", f"Please check the run inputs.\n\n{exc}")
            return

        config_overrides["algorithm_params"] = dict(DEFAULT_EXPERIMENT_CONFIG["algorithm_params"])
        config_overrides["algorithm_params"]["mutation_rate"] = mutation_rate

        self.live_generations_df = pd.DataFrame(
            columns=[
                "generation_id",
                "mean_qty_aggression",
                "mean_signal_aggression",
                "mean_threshold",
                "mean_signal_clip",
                "mean_info_param_parameterised_informed",
                "mean_info_param_zi",
                "mean_wealth_parameterised_informed",
                "mean_wealth_zi",
            ]
        )
        self.live_strategy_generation_df = pd.DataFrame()
        self.live_strategy_round_df = pd.DataFrame()
        self._clear_round_plots()
        self.tabs.setCurrentWidget(self.dashboard_tab)

        self.btn_start_run.setEnabled(False)
        self.run_status_label.setText("Starting experiment...")

        self.run_worker = ExperimentRunnerWorker(config_overrides)
        self.run_worker.progress.connect(self._handle_run_progress)
        self.run_worker.completed.connect(self._handle_run_completed)
        self.run_worker.error.connect(self._handle_run_error)
        self.run_worker.finished.connect(self._run_worker_finished)
        self.run_worker.start()

        self.btn_stop_run.setEnabled(True)

    def _worker_finished(self):
        self.btn_refresh.setEnabled(True)
        self.worker = None

    def _run_worker_finished(self):
        self.btn_start_run.setEnabled(True)
        self.btn_stop_run.setEnabled(False)
        self.run_worker = None

    def _show_error(self, message):
        self.status_label.setText("Load failed.")
        if "used by another process" in message.lower():
            message = (
                f"{message}\n\n"
                "Close any running main.py or other DuckDB session using this file, "
                "then press Refresh Database again."
            )
        QMessageBox.critical(self, "Database Load Error", message)

    def _handle_run_progress(self, payload):
        event = payload.get("event")
        if event == "generation_started":
            self._current_experiment_id = payload.get("experiment_id")
            self.run_status_label.setText(
                f"Running generation {payload['generation_id']} of {payload['n_generations']}..."
            )
            return

        if event == "generation_completed":
            self._current_experiment_id = payload.get("experiment_id")
            self._current_generation_id = payload.get("generation_id")
            metrics = payload.get("generation_metrics", {})
            if metrics:
                self.live_generations_df = pd.concat(
                    [self.live_generations_df, pd.DataFrame([metrics])],
                    ignore_index=True,
                )
                self.live_generations_df = (
                    self.live_generations_df.drop_duplicates(subset=["generation_id"], keep="last")
                    .sort_values("generation_id")
                    .reset_index(drop=True)
                )
                self._update_parameter_plot(self.live_generations_df)
                self._update_wealth_plot_from_generation_metrics(self.live_generations_df)
                self._update_mean_info_param_plot(self.live_generations_df)

            self._update_market_plot(pd.DataFrame(payload.get("market_summary", [])))
            self._update_profit_plot(pd.DataFrame(payload.get("strategy_profit_per_round", [])))
            self._update_volume_share_plot(pd.DataFrame(payload.get("volume_share_per_round", [])))
            round_df = pd.DataFrame(payload.get("strategy_performance_round", []))
            if not round_df.empty:
                self.live_strategy_round_df = round_df
                self._update_metric_grid(
                    self.live_strategy_round_df,
                    index_col="round_number",
                    curve_store=self.performance_round_curves,
                )

            generation_df = pd.DataFrame(payload.get("strategy_performance_generation", []))
            if not generation_df.empty:
                wealth_generation_df = pd.DataFrame(
                    [
                        {
                            "generation_id": payload["generation_id"],
                            "strategy_type": "parameterised_informed",
                            "avg_wealth_per_gen": metrics.get("mean_wealth_parameterised_informed"),
                        },
                        {
                            "generation_id": payload["generation_id"],
                            "strategy_type": "zi",
                            "avg_wealth_per_gen": metrics.get("mean_wealth_zi"),
                        },
                    ]
                )
                generation_df = generation_df.merge(
                    wealth_generation_df,
                    on=["generation_id", "strategy_type"],
                    how="left",
                )
                self.live_strategy_generation_df = pd.concat(
                    [self.live_strategy_generation_df, generation_df],
                    ignore_index=True,
                )
                self.live_strategy_generation_df = (
                    self.live_strategy_generation_df.drop_duplicates(
                        subset=["generation_id", "strategy_type"],
                        keep="last",
                    )
                    .sort_values(["generation_id", "strategy_type"])
                    .reset_index(drop=True)
                )
                self._update_metric_grid(
                    self.live_strategy_generation_df,
                    index_col="generation_id",
                    curve_store=self.performance_generation_curves,
                )

            self.run_status_label.setText(
                f"Completed generation {payload['generation_id']} of {payload['n_generations']}."
            )
            self.summary_label.setText(
                "\n".join(
                    [
                        f"Experiment ID: {self._current_experiment_id}",
                        f"Live generation: {self._current_generation_id}",
                        f"Completed generations: {len(self.live_generations_df)}",
                    ]
                )
            )
            return

        if event == "experiment_completed":
            self.run_status_label.setText(f"Experiment {payload['experiment_id']} completed.")

    def _handle_run_completed(self, result):
        self._current_experiment_id = result["experiment_id"]
        self._current_generation_id = None
        self.run_status_label.setText(
            f"Finished experiment {result['experiment_id']}."
        )
        self.refresh_data()

    def stop_run(self):
        if self.run_worker is not None:
            self.run_worker.terminate()
            self.run_status_label.setText("Run stopped by user.")
            self.btn_stop_run.setEnabled(False)

    def _handle_run_error(self, message):
        self.run_status_label.setText("Run failed.")
        QMessageBox.critical(self, "Run Error", message)

    def _apply_payload(self, payload):
        self._last_payload = payload
        self._current_experiment_id = payload["selected_experiment_id"]
        self._current_generation_id = payload["selected_generation_id"]

        self._populate_experiment_combo(payload["experiments"])
        self._populate_generation_combo(payload["generations"])

        self._populate_sql_tab(payload.get("sql_objects", {}))

        self._update_summary(payload)
        self._update_parameter_plot(payload["generations"])
        self._update_wealth_plot(payload["wealth_history"])
        self._update_mean_info_param_plot(payload["mean_info_param"])
        self._update_market_plot(payload["market_summary"])
        self._update_profit_plot(payload["strategy_profit_round"])
        self._update_volume_share_plot(payload["volume_share_round"])
        self._update_metric_grid(
            self._build_strategy_generation_plot_df(
                payload["strategy_generation"],
                payload["wealth_history"],
            ),
            index_col="generation_id",
            curve_store=self.performance_generation_curves,
        )
        self._update_metric_grid(
            payload["strategy_profit_round"],
            index_col="round_number",
            curve_store=self.performance_round_curves,
        )
        self._update_metric_grid(
            payload["agent_strategy_evolution"],
            index_col="generation_id",
            curve_store=self.agent_generation_curves,
        )
        self._update_agent_info_param_plot(payload["agent_info_param_history"])
        self._update_agent_generation_param_plot(
            payload["agent_strategy_param_history"],
            "qty_aggression",
            self.agent_qty_aggression_plot,
        )
        self._update_agent_generation_param_plot(
            payload["agent_strategy_param_history"],
            "signal_aggression",
            self.agent_signal_aggression_plot,
        )
        self._update_agent_generation_param_plot(
            payload["agent_strategy_param_history"],
            "threshold",
            self.agent_threshold_plot,
        )
        self._update_agent_generation_param_plot(
            payload["agent_strategy_param_history"],
            "signal_clip",
            self.agent_signal_clip_plot,
        )
        self._update_agent_metric_plot(payload["agent_profit_loss"], "profit_loss")
        self._update_agent_metric_plot(payload["agent_fill_rate"], "fill_rate")
        self._update_agent_metric_plot(payload["agent_inventory_risk"], "inventory_risk")
        self._update_agent_metric_plot(payload["agent_inventory_turnover"], "inventory_turnover")
        self._update_agent_metric_plot(payload["agent_relative_performance"], "relative_profit_loss")
        self._update_agent_metric_plot(payload["agent_signal_accuracy"], "signal_accuracy")
        self._update_agent_metric_plot(payload["agent_volume_share"], "volume_share")
        self._update_agent_metric_plot(payload["agent_avg_trade_size"], "avg_trade_size")
        self._update_agent_metric_plot(payload["agent_aggressiveness_spread"], "aggressiveness")
        self._update_agent_metric_plot(payload["agent_aggressiveness_spread"], "market_spread")
        self._update_agent_metric_plot(payload["agent_behavior_change"], "aggressiveness_change")
        self._update_agent_metric_plot(payload["agent_behavior_change"], "order_qty_change")
        self._update_agent_metric_plot(payload["agent_behavior_change"], "inventory_change")
        self._update_agent_metric_plot(
            payload["agent_execution_price_deviation"],
            "execution_price_deviation",
        )

        self.status_label.setText("Loaded data from DuckDB.")

    def _populate_experiment_combo(self, experiments_df):
        self._suppress_selection_signals = True
        self.combo_experiment.clear()
        for _, row in experiments_df.iterrows():
            self.combo_experiment.addItem(
                f"{row['experiment_name']} | {row['experiment_id']}",
                row["experiment_id"],
            )
        if self._current_experiment_id is not None:
            index = self.combo_experiment.findData(self._current_experiment_id)
            if index >= 0:
                self.combo_experiment.setCurrentIndex(index)
        self._suppress_selection_signals = False

    def _populate_sql_tab(self, sql_objects):
        while self.sql_tab.count() > 0:
            widget = self.sql_tab.widget(0)
            self.sql_tab.removeTab(0)
            if widget is not None:
                widget.deleteLater()

        self.sql_tables = {}
        self.table_experiments = None
        self.table_generations = None
        self.table_population = None
        self.table_market = None
        self.table_agent_round = None
        self.table_strategy = None

        legacy_names = {
            "experiments": "table_experiments",
            "generations": "table_generations",
            "agent_population": "table_population",
            "market_round": "table_market",
            "agent_round": "table_agent_round",
            "strategy_performance_per_generation": "table_strategy",
        }

        for object_name, object_meta in sql_objects.items():
            table = QTableWidget()
            self._populate_table(table, object_meta["data"])
            object_kind = "View" if object_meta["type"] == "VIEW" else "Table"
            display_name = object_meta.get("display_name", _humanize_sql_object_name(object_name))
            preview_rows = int(object_meta.get("preview_rows", 0))
            total_rows = int(object_meta.get("row_count", preview_rows))
            if total_rows > preview_rows:
                label = f"{display_name} {object_kind} ({preview_rows}/{total_rows})"
            else:
                label = f"{display_name} {object_kind} ({total_rows})"
            self.sql_tab.addTab(table, label)
            self.sql_tables[object_name] = table

            legacy_attr = legacy_names.get(object_name)
            if legacy_attr is not None:
                setattr(self, legacy_attr, table)

    def _populate_generation_combo(self, generations_df):
        self._suppress_selection_signals = True
        self.combo_generation.clear()
        for _, row in generations_df.iterrows():
            self.combo_generation.addItem(
                f"Generation {int(row['generation_id'])} | {row['generation_status']}",
                int(row["generation_id"]),
            )
        if self._current_generation_id is not None:
            index = self.combo_generation.findData(int(self._current_generation_id))
            if index >= 0:
                self.combo_generation.setCurrentIndex(index)

        if generations_df.empty:
            self.generation_slider.setEnabled(False)
            self.generation_slider.setMinimum(1)
            self.generation_slider.setMaximum(1)
            self.generation_slider.setValue(1)
            self.generation_slider_label.setText("Generation slider unavailable")
        else:
            min_generation = int(generations_df["generation_id"].min())
            max_generation = int(generations_df["generation_id"].max())
            slider_value = int(self._current_generation_id) if self._current_generation_id is not None else max_generation
            self.generation_slider.setEnabled(True)
            self.generation_slider.setMinimum(min_generation)
            self.generation_slider.setMaximum(max_generation)
            self.generation_slider.setValue(slider_value)
            self.generation_slider_label.setText(
                f"Generation {slider_value} of {max_generation}"
            )
        self._suppress_selection_signals = False

    def _on_experiment_changed(self, index):
        if self._suppress_selection_signals or index < 0:
            return
        self._current_experiment_id = self.combo_experiment.itemData(index)
        self._current_generation_id = None
        self.refresh_data()

    def _on_generation_changed(self, index):
        if self._suppress_selection_signals or index < 0:
            return
        self._current_generation_id = self.combo_generation.itemData(index)
        self.generation_slider_label.setText(
            f"Generation {int(self._current_generation_id)} of {self.generation_slider.maximum()}"
        )
        if self.generation_slider.value() != int(self._current_generation_id):
            self._suppress_selection_signals = True
            self.generation_slider.setValue(int(self._current_generation_id))
            self._suppress_selection_signals = False
        self.refresh_data()

    def _on_generation_slider_changed(self, value):
        if self._suppress_selection_signals or not self.generation_slider.isEnabled():
            return
        self.generation_slider_label.setText(
            f"Generation {int(value)} of {self.generation_slider.maximum()}"
        )
        self._pending_generation_id = int(value)
        self._generation_slider_timer.start()

    def _apply_debounced_generation_change(self):
        if self._pending_generation_id is None:
            return
        index = self.combo_generation.findData(int(self._pending_generation_id))
        if index >= 0:
            self.combo_generation.setCurrentIndex(index)

    def _on_smoothing_changed(self, value):
        self._pending_smoothing_window = int(value)
        if self._pending_smoothing_window <= 1:
            self.smoothing_slider_label.setText("Graph smoothing: 1 (off)")
        else:
            self.smoothing_slider_label.setText(
                f"Graph smoothing: {self._pending_smoothing_window}-point rolling mean"
            )
        self._smoothing_slider_timer.start()

    def _apply_debounced_smoothing_change(self):
        self.smoothing_window = int(self._pending_smoothing_window)
        self._refresh_plots_only()

    def _format_plot_title(self, base_title):
        return (
            f"<span style='color: {GRAPH_FOREGROUND};'>"
            f"{base_title} | smoothing {self.smoothing_window}"
            f"</span>"
        )

    def _update_all_plot_titles(self):
        self.plot_params.setTitle(self._format_plot_title(self.plot_params_title))
        self.plot_wealth.setTitle(self._format_plot_title(self.plot_wealth_title))
        self.plot_info_param.setTitle(self._format_plot_title(self.plot_info_param_title))
        self.plot_market.setTitle(self._format_plot_title(self.plot_market_title))
        self.plot_profit.setTitle(self._format_plot_title(self.plot_profit_title))
        self.plot_volume_share.setTitle(self._format_plot_title(self.plot_volume_share_title))

        for plot, base_title in self.performance_generation_plots.values():
            plot.setTitle(self._format_plot_title(base_title))
        for plot, base_title in self.performance_round_plots.values():
            plot.setTitle(self._format_plot_title(base_title))
        for plot, base_title in self.agent_generation_plots.values():
            plot.setTitle(self._format_plot_title(base_title))
        self.agent_info_param_plot.setTitle(
            self._format_plot_title("Info_Param per Agent")
        )
        self.agent_qty_aggression_plot.setTitle(
            self._format_plot_title("Qty_Aggression per Agent")
        )
        self.agent_signal_aggression_plot.setTitle(
            self._format_plot_title("Signal_Aggression per Agent")
        )
        self.agent_threshold_plot.setTitle(
            self._format_plot_title("Threshold per Agent")
        )
        self.agent_signal_clip_plot.setTitle(
            self._format_plot_title("Signal_Clip per Agent")
        )
        for plot, base_title in self.agent_round_plots.values():
            plot.setTitle(self._format_plot_title(base_title))

    def _smooth_series(self, values):
        if self.smoothing_window <= 1:
            return list(values)
        series = pd.Series(values, dtype=float)
        return series.rolling(window=self.smoothing_window, min_periods=1).mean().tolist()

    def _is_strategy_visible(self, strategy_type):
        if strategy_type == "parameterised_informed":
            return self.checkbox_show_parameterised.isChecked()
        if strategy_type == "zi":
            return self.checkbox_show_zi.isChecked()
        return True

    def _refresh_plots_only(self):
        if self._last_payload is None:
            return
        self._update_all_plot_titles()
        self._update_parameter_plot(self._last_payload["generations"])
        self._update_wealth_plot(self._last_payload["wealth_history"])
        self._update_mean_info_param_plot(self._last_payload["mean_info_param"])
        self._update_market_plot(self._last_payload["market_summary"])
        self._update_profit_plot(self._last_payload["strategy_profit_round"])
        self._update_volume_share_plot(self._last_payload["volume_share_round"])
        self._update_metric_grid(
            self._build_strategy_generation_plot_df(
                self._last_payload["strategy_generation"],
                self._last_payload["wealth_history"],
            ),
            index_col="generation_id",
            curve_store=self.performance_generation_curves,
        )
        self._update_metric_grid(
            self._last_payload["strategy_profit_round"],
            index_col="round_number",
            curve_store=self.performance_round_curves,
        )
        self._update_metric_grid(
            self._last_payload["agent_strategy_evolution"],
            index_col="generation_id",
            curve_store=self.agent_generation_curves,
        )
        self._update_agent_info_param_plot(self._last_payload["agent_info_param_history"])
        self._update_agent_generation_param_plot(
            self._last_payload["agent_strategy_param_history"],
            "qty_aggression",
            self.agent_qty_aggression_plot,
        )
        self._update_agent_generation_param_plot(
            self._last_payload["agent_strategy_param_history"],
            "signal_aggression",
            self.agent_signal_aggression_plot,
        )
        self._update_agent_generation_param_plot(
            self._last_payload["agent_strategy_param_history"],
            "threshold",
            self.agent_threshold_plot,
        )
        self._update_agent_generation_param_plot(
            self._last_payload["agent_strategy_param_history"],
            "signal_clip",
            self.agent_signal_clip_plot,
        )
        self._update_agent_metric_plot(self._last_payload["agent_profit_loss"], "profit_loss")
        self._update_agent_metric_plot(self._last_payload["agent_fill_rate"], "fill_rate")
        self._update_agent_metric_plot(self._last_payload["agent_inventory_risk"], "inventory_risk")
        self._update_agent_metric_plot(self._last_payload["agent_inventory_turnover"], "inventory_turnover")
        self._update_agent_metric_plot(self._last_payload["agent_relative_performance"], "relative_profit_loss")
        self._update_agent_metric_plot(self._last_payload["agent_signal_accuracy"], "signal_accuracy")
        self._update_agent_metric_plot(self._last_payload["agent_volume_share"], "volume_share")
        self._update_agent_metric_plot(self._last_payload["agent_avg_trade_size"], "avg_trade_size")
        self._update_agent_metric_plot(self._last_payload["agent_aggressiveness_spread"], "aggressiveness")
        self._update_agent_metric_plot(self._last_payload["agent_aggressiveness_spread"], "market_spread")
        self._update_agent_metric_plot(self._last_payload["agent_behavior_change"], "aggressiveness_change")
        self._update_agent_metric_plot(self._last_payload["agent_behavior_change"], "order_qty_change")
        self._update_agent_metric_plot(self._last_payload["agent_behavior_change"], "inventory_change")
        self._update_agent_metric_plot(
            self._last_payload["agent_execution_price_deviation"],
            "execution_price_deviation",
        )

    def _update_summary(self, payload):
        experiments_df = payload["experiments"]
        generations_df = payload["generations"]
        if experiments_df.empty or self._current_experiment_id is None:
            self.summary_label.setText("No experiment data available.")
            return

        selected_row = experiments_df.loc[experiments_df["experiment_id"] == self._current_experiment_id]
        if selected_row.empty:
            self.summary_label.setText("No experiment selected.")
            return

        experiment = selected_row.iloc[0]
        completed_gens = 0
        if not generations_df.empty:
            completed_gens = int((generations_df["generation_status"] == "COMPLETED").sum())

        generation_text = (
            f"Selected generation: {self._current_generation_id}"
            if self._current_generation_id is not None
            else "Selected generation: none"
        )
        self.summary_label.setText(
            "\n".join(
                [
                    f"Experiment: {experiment['experiment_name']}",
                    f"Experiment ID: {experiment['experiment_id']}",
                    f"Type: {experiment['experiment_type']}",
                    f"Generations completed: {completed_gens}/{experiment['n_generations']}",
                    f"Rounds per generation: {experiment['n_rounds']}",
                    f"Total agents: {experiment['n_agents']}",
                    generation_text,
                ]
            )
        )

    def _update_parameter_plot(self, generations_df):
        if generations_df.empty or not self._is_strategy_visible("parameterised_informed"):
            self.line_qty.setData([], [])
            self.line_signal.setData([], [])
            self.line_threshold.setData([], [])
            self.line_clip.setData([], [])
            return

        x = generations_df["generation_id"].tolist()
        self.line_qty.setData(x, self._smooth_series(generations_df["mean_qty_aggression"].tolist()))
        self.line_signal.setData(x, self._smooth_series(generations_df["mean_signal_aggression"].tolist()))
        self.line_threshold.setData(x, self._smooth_series(generations_df["mean_threshold"].tolist()))
        self.line_clip.setData(x, self._smooth_series(generations_df["mean_signal_clip"].tolist()))

    def _update_wealth_plot(self, wealth_history_df):
        if wealth_history_df.empty:
            self.line_informed_wealth.setData([], [])
            self.line_zi_wealth.setData([], [])
            return

        pivot = wealth_history_df.pivot(index="generation_id", columns="strategy_type", values="mean_wealth").sort_index()
        x = pivot.index.tolist()
        informed = pivot.get("parameterised_informed", pd.Series(dtype=float)).reindex(pivot.index)
        zi = pivot.get("zi", pd.Series(dtype=float)).reindex(pivot.index)
        self.line_informed_wealth.setData(
            x if self._is_strategy_visible("parameterised_informed") else [],
            self._smooth_series(informed.tolist()) if self._is_strategy_visible("parameterised_informed") else [],
        )
        self.line_zi_wealth.setData(
            x if self._is_strategy_visible("zi") else [],
            self._smooth_series(zi.tolist()) if self._is_strategy_visible("zi") else [],
        )

    def _update_wealth_plot_from_generation_metrics(self, metrics_df):
        x = metrics_df["generation_id"].tolist()
        self.line_informed_wealth.setData(
            x if self._is_strategy_visible("parameterised_informed") else [],
            self._smooth_series(metrics_df["mean_wealth_parameterised_informed"].tolist())
            if self._is_strategy_visible("parameterised_informed")
            else [],
        )
        self.line_zi_wealth.setData(
            x if self._is_strategy_visible("zi") else [],
            self._smooth_series(metrics_df["mean_wealth_zi"].tolist()) if self._is_strategy_visible("zi") else [],
        )

    def _update_mean_info_param_plot(self, mean_info_param_df):
        if mean_info_param_df.empty:
            self.line_info_param_informed.setData([], [])
            self.line_info_param_zi.setData([], [])
            return

        if {"generation_id", "strategy_type", "mean_info_param"}.issubset(mean_info_param_df.columns):
            pivot = (
                mean_info_param_df.pivot(
                    index="generation_id",
                    columns="strategy_type",
                    values="mean_info_param",
                )
                .sort_index()
            )
            x = pivot.index.tolist()
            informed = pivot.get("parameterised_informed", pd.Series(dtype=float)).reindex(pivot.index)
            zi = pivot.get("zi", pd.Series(dtype=float)).reindex(pivot.index)
        else:
            sorted_df = mean_info_param_df.sort_values("generation_id")
            x = sorted_df["generation_id"].tolist()
            informed = sorted_df.get(
                "mean_info_param_parameterised_informed",
                pd.Series([float("nan")] * len(sorted_df)),
            )
            zi = sorted_df.get(
                "mean_info_param_zi",
                pd.Series([float("nan")] * len(sorted_df)),
            )

        self.line_info_param_informed.setData(
            x if self._is_strategy_visible("parameterised_informed") else [],
            self._smooth_series(informed.tolist()) if self._is_strategy_visible("parameterised_informed") else [],
        )
        self.line_info_param_zi.setData(
            x if self._is_strategy_visible("zi") else [],
            self._smooth_series(zi.tolist()) if self._is_strategy_visible("zi") else [],
        )

    def _update_market_plot(self, market_summary_df):
        if market_summary_df.empty:
            self.line_max_bid.setData([], [])
            self.line_max_sell.setData([], [])
            self.line_min_bid.setData([], [])
            self.line_min_sell.setData([], [])
            self.line_bid_q2.setData([], [])
            self.line_ask_q3.setData([], [])
            self.line_fundamental_price.setData([], [])
            self.line_mid_price.setData([], [])
            return

        x = market_summary_df["round_number"].tolist()
        self.line_max_bid.setData(x, self._smooth_series(market_summary_df["max_bid"].tolist()))
        self.line_max_sell.setData(x, self._smooth_series(market_summary_df["max_sell"].tolist()))
        self.line_min_bid.setData(x, self._smooth_series(market_summary_df["min_bid"].tolist()))
        self.line_min_sell.setData(x, self._smooth_series(market_summary_df["min_sell"].tolist()))
        self.line_bid_q2.setData(x, self._smooth_series(market_summary_df["bid_price_q2"].tolist()))
        self.line_ask_q3.setData(x, self._smooth_series(market_summary_df["ask_price_q3"].tolist()))
        self.line_fundamental_price.setData(
            x, self._smooth_series(market_summary_df["fundamental_price"].tolist())
        )
        self.line_mid_price.setData(x, self._smooth_series(market_summary_df["mid_price"].tolist()))

    def _update_profit_plot(self, strategy_profit_df):
        self._update_strategy_pair_plot(
            df=strategy_profit_df,
            value_col="avg_profit_loss",
            informed_line=self.line_profit_informed,
            zi_line=self.line_profit_zi,
        )

    def _update_volume_share_plot(self, volume_share_df):
        self._update_strategy_pair_plot(
            df=volume_share_df,
            value_col="avg_volume_share",
            informed_line=self.line_volume_informed,
            zi_line=self.line_volume_zi,
        )

    def _update_strategy_pair_plot(self, df, value_col, informed_line, zi_line):
        if df.empty:
            informed_line.setData([], [])
            zi_line.setData([], [])
            return

        pivot = df.pivot(index="round_number", columns="strategy_type", values=value_col).sort_index()
        x = pivot.index.tolist()
        informed = pivot.get("parameterised_informed", pd.Series(dtype=float)).reindex(pivot.index)
        zi = pivot.get("zi", pd.Series(dtype=float)).reindex(pivot.index)
        informed_line.setData(
            x if self._is_strategy_visible("parameterised_informed") else [],
            self._smooth_series(informed.tolist()) if self._is_strategy_visible("parameterised_informed") else [],
        )
        zi_line.setData(
            x if self._is_strategy_visible("zi") else [],
            self._smooth_series(zi.tolist()) if self._is_strategy_visible("zi") else [],
        )

    def _build_strategy_generation_plot_df(self, strategy_generation_df, wealth_history_df):
        merged_df = strategy_generation_df.copy()

        if not wealth_history_df.empty:
            wealth_df = wealth_history_df.rename(columns={"mean_wealth": "avg_wealth_per_gen"}).copy()
            wealth_df = wealth_df[["generation_id", "strategy_type", "avg_wealth_per_gen"]]
            if merged_df.empty:
                merged_df = wealth_df
            else:
                merged_df = merged_df.merge(
                    wealth_df,
                    on=["generation_id", "strategy_type"],
                    how="outer",
                )

        if merged_df.empty:
            return merged_df

        return merged_df.sort_values(["generation_id", "strategy_type"]).reset_index(drop=True)

    def _clear_round_plots(self):
        self._update_market_plot(pd.DataFrame())
        self._update_profit_plot(pd.DataFrame())
        self._update_volume_share_plot(pd.DataFrame())

    def _update_metric_grid(self, df, index_col, curve_store):
        if df.empty:
            for curves in curve_store.values():
                for curve in curves.values():
                    curve.setData([], [])
            return

        for metric_key, curves in curve_store.items():
            if metric_key not in df.columns:
                for curve in curves.values():
                    curve.setData([], [])
                continue

            pivot = df.pivot(index=index_col, columns="strategy_type", values=metric_key).sort_index()
            x = pivot.index.tolist()
            for strategy_type, curve in curves.items():
                series = pivot.get(strategy_type, pd.Series(dtype=float)).reindex(pivot.index)
                if self._is_strategy_visible(strategy_type):
                    curve.setData(x, self._smooth_series(series.tolist()))
                else:
                    curve.setData([], [])

    def _update_agent_metric_plot(self, df, metric_key):
        plot = self.agent_round_plots[metric_key][0]
        plot.clear()
        if df.empty or metric_key not in df.columns:
            return

        if "agent_id" not in df.columns or "strategy_type" not in df.columns:
            return

        for _, agent_df in df.sort_values(["agent_id", "round_number"]).groupby("agent_id"):
            strategy_type = str(agent_df["strategy_type"].iloc[0])
            if not self._is_strategy_visible(strategy_type):
                continue
            color = AGENT_PERFORMANCE_COLORS.get(strategy_type, (120, 120, 120))
            plot.plot(
                x=agent_df["round_number"].tolist(),
                y=self._smooth_series(agent_df[metric_key].tolist()),
                pen=pg.mkPen(color, width=1),
            )

    def _update_agent_info_param_plot(self, info_param_history_df):
        self._update_agent_generation_line_plot(
            info_param_history_df,
            "info_param",
            self.agent_info_param_plot,
        )

    def _update_agent_generation_param_plot(self, param_history_df, param_name, plot):
        self._update_agent_generation_line_plot(param_history_df, param_name, plot)

    def _update_agent_generation_line_plot(self, history_df, value_col, plot):
        plot.clear()
        if history_df.empty or value_col not in history_df.columns:
            return

        required_cols = {"generation_id", "agent_id", "strategy_type", value_col}
        if not required_cols.issubset(history_df.columns):
            return

        filtered_df = history_df[
            history_df["strategy_type"] == "parameterised_informed"
        ].copy()
        if filtered_df.empty or not self._is_strategy_visible("parameterised_informed"):
            return

        filtered_df = filtered_df.dropna(subset=[value_col]).sort_values(["agent_id", "generation_id"])
        if filtered_df.empty:
            return

        color = AGENT_PERFORMANCE_COLORS["parameterised_informed"]
        for _, agent_df in filtered_df.groupby("agent_id"):
            plot.plot(
                x=agent_df["generation_id"].tolist(),
                y=self._smooth_series(agent_df[value_col].astype(float).tolist()),
                pen=pg.mkPen(color, width=1),
                symbol="o",
                symbolSize=4,
                symbolBrush=color,
                symbolPen=color,
            )

    def _populate_table(self, table, df):
        table.clear()
        table.setColumnCount(len(df.columns))
        table.setRowCount(len(df.index))
        table.setHorizontalHeaderLabels([str(col) for col in df.columns])

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, value in enumerate(row):
                if pd.isna(value):
                    display_value = ""
                elif isinstance(value, float):
                    display_value = f"{value:.6f}"
                else:
                    display_value = str(value)

                item = QTableWidgetItem(display_value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_idx, col_idx, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CommandCenter()
    window.show()
    sys.exit(app.exec())
