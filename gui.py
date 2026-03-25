"""
PySide6 GUI for the agent-based market simulation: experiment configuration,
live generation monitoring, database browsing, and multi-experiment comparison.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtGui import QColor
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
    QListWidget,
    QListWidgetItem,
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

from constants import COMPARISON_PARAM_SPECS, WEALTH_INFORMED_COL, WEALTH_ZI_COL
from database_creation import create_database
from main import DEFAULT_EXPERIMENT_CONFIG, run_experiment


DEFAULT_DB_PATH = "experiment_results.duckdb"
SQL_TAB_PREVIEW_ROWS = 1000
GRAPH_BACKGROUND = "#2b2b2b"
GRAPH_FOREGROUND = "#ffffff"
APP_THEMES = {
    "dark": {
        "window_bg": "#171b21",
        "surface_bg": "#1f252d",
        "surface_alt_bg": "#262d37",
        "input_bg": "#11161c",
        "border": "#3b4654",
        "text": "#f3f5f7",
        "muted_text": "#b8c2cf",
        "selection_bg": "#2b6cb0",
        "selection_text": "#ffffff",
        "plot_background": "#2b2b2b",
        "plot_foreground": "#ffffff",
        "plot_reference": (255, 255, 255, 100),
        "plot_outline": (255, 255, 255),
        "plot_mid_price": (255, 255, 255),
        "refresh_button": "#2e8b57",
        "start_button": "#1f5fa5",
        "batch_button": "#7b4bb7",
        "stop_button": "#a52a2a",
        "compare_button": "#6a5acd",
    },
    "light": {
        "window_bg": "#f3f5f8",
        "surface_bg": "#ffffff",
        "surface_alt_bg": "#e9eef5",
        "input_bg": "#ffffff",
        "border": "#c3cfdb",
        "text": "#17212b",
        "muted_text": "#4a5563",
        "selection_bg": "#2b6cb0",
        "selection_text": "#ffffff",
        "plot_background": "#ffffff",
        "plot_foreground": "#17212b",
        "plot_reference": (23, 33, 43, 110),
        "plot_outline": (23, 33, 43),
        "plot_mid_price": (23, 33, 43),
        "refresh_button": "#2f855a",
        "start_button": "#2b6cb0",
        "batch_button": "#805ad5",
        "stop_button": "#c53030",
        "compare_button": "#5a67d8",
    },
}
STRATEGY_COLORS = {
    "zi": (255, 140, 0),
    "parameterised_informed": (30, 144, 255),
}
COMPARISON_WEALTH_DIFF_SPEC = (
    "wealth_difference",
    "Informed Wealth - ZI Wealth",
)
PLOT_MATH_NOTES = {
    "Mean Strategy Parameters Across Generations": "q<sub>g</sub>, s<sub>g</sub>",
    "Mean Wealth by Strategy": "W&#772;<sub>g</sub><sup>(s)</sup> = avg<sub>i</sub>[cash + inv p<sub>T</sub>]",
    "Mean Info_Param by Strategy": "&theta;&#772;<sub>g</sub><sup>(s)</sup> = avg<sub>i</sub>[&theta;<sub>i</sub>]",
    "Market Summary by Round": "bid/ask extrema, F<sub>t</sub>, mid<sub>t</sub>",
    "Average Profit per Round: ZI vs Parameterised": "&pi;&#772;<sub>t</sub><sup>(s)</sup> = avg<sub>i</sub> [&Delta;W<sub>i,t</sub>]",
    "Average Agent Volume Share per Round": "v&#772;<sub>t</sub><sup>(s)</sup> = avg<sub>i</sub>[q<sub>i,t</sub>/V<sub>t</sub>]",
    "Qty Aggression": "q&#772;<sub>g</sub> = avg<sub>i</sub>[q<sub>i</sub>]",
    "Signal Aggression": "s&#772;<sub>g</sub> = avg<sub>i</sub>[s<sub>i</sub>]",
    "Info Param (informed)": "&theta;&#772;<sub>g</sub><sup>I</sup> = avg<sub>i</sub>[&theta;<sub>i</sub> | I]",
    "Mean Info Param (informed)": "&theta;&#772;<sub>g</sub><sup>I</sup> = avg<sub>i</sub>[&theta;<sub>i</sub> | I]",
    "Qty Aggression Std Dev": "&sigma;<sub>q,g</sub> = std<sub>i</sub>[q<sub>i</sub>]",
    "Signal Aggression Std Dev": "&sigma;<sub>s,g</sub> = std<sub>i</sub>[s<sub>i</sub>]",
    "Info Param Std Dev": "&sigma;<sub>&theta;,g</sub> = std<sub>i</sub>[&theta;<sub>i</sub> | I]",
    "Informed Wealth - ZI Wealth": "&Delta;W<sub>g</sub> = W&#772;<sub>g</sub><sup>I</sup> - W&#772;<sub>g</sub><sup>ZI</sup>",
    "Limit Order Book Snapshot": "Q<sub>t</sub>(p) = queued qty at price p",
    "Candlestick View": "(O<sub>t</sub>, H<sub>t</sub>, L<sub>t</sub>, C<sub>t</sub>)",
    "Trade Network": "E<sub>t</sub>(i,j) = notional flow",
    "Order Imbalance and Spread": "I<sub>t</sub> = (D<sub>t</sub> - S<sub>t</sub>)/(D<sub>t</sub> + S<sub>t</sub>)",
    "Participation and Volume": "V<sub>t</sub>, N<sub>trades,t</sub>, N<sub>active,t</sub>",
    "Profit Change vs Prev Gen": "&Delta;&pi;<sub>g</sub><sup>(s)</sup> = &pi;<sub>g</sub><sup>(s)</sup> - &pi;<sub>g-1</sub><sup>(s)</sup>",
    "Info_Param per Agent": "&theta;<sub>i,g</sub>",
    "Qty_Aggression per Agent": "q<sub>i,g</sub>",
    "Signal_Aggression per Agent": "s<sub>i,g</sub>",
}
PLOT_BRIEF_NOTES = {
    "Mean Strategy Parameters Across Generations": "Shows how the learned strategy aggressiveness parameters evolve generation by generation.",
    "Mean Wealth by Strategy": "Compares average end-of-generation wealth for informed and zero-intelligence traders.",
    "Mean Info_Param by Strategy": "Tracks the average information parameter carried by each strategy group across generations.",
    "Market Summary by Round": "Summarises how bids, asks, the fundamental value, and the mid price move through a generation.",
    "Average Profit per Round: ZI vs Parameterised": "Shows round-by-round average profit so you can compare trading performance within a generation.",
    "Average Agent Volume Share per Round": "Measures how much of each round's executed volume is supplied by each strategy group.",
    "Qty Aggression": "Shows the cross-run mean quantity aggressiveness over generations.",
    "Signal Aggression": "Shows the cross-run mean signal aggressiveness over generations.",
    "Info Param (informed)": "Shows how the informed traders' average information parameter changes over generations.",
    "Mean Info Param (informed)": "Shows how the informed traders' average information parameter changes over generations.",
    "Qty Aggression Std Dev": "Shows how dispersed quantity aggressiveness is within the informed population.",
    "Signal Aggression Std Dev": "Shows how dispersed signal aggressiveness is within the informed population.",
    "Info Param Std Dev": "Shows how spread out informed traders are in their information parameter values.",
    "Informed Wealth - ZI Wealth": "Positive values mean informed traders outperform ZI traders on average; negative values mean the reverse.",
    "Limit Order Book Snapshot": "Displays queued demand and supply at each price for the selected round.",
    "Candlestick View": "Shows a compact open-high-low-close style summary of market prices by round.",
    "Trade Network": "Maps who traded with whom, with edge thickness reflecting notional flow.",
    "Order Imbalance and Spread": "Compares buy-sell pressure with the prevailing quoted spread over time.",
    "Participation and Volume": "Shows how trading activity changes through the generation using volume, trades, and active participants.",
    "Profit Change vs Prev Gen": "Shows whether each strategy's generation-level profit is improving or deteriorating versus the previous generation.",
    "Info_Param per Agent": "Tracks each informed agent's information parameter across generations.",
    "Qty_Aggression per Agent": "Tracks each informed agent's quantity aggressiveness across generations.",
    "Signal_Aggression per Agent": "Tracks each informed agent's signal aggressiveness across generations.",
}
COMPARISON_LINE_COLORS = [
    (255, 140, 0),
    (30, 144, 255),
    (220, 20, 60),
    (46, 139, 87),
    (218, 165, 32),
    (186, 85, 211),
    (0, 206, 209),
    (205, 92, 92),
]
SWEEP_COMPARISON_OUTPUT_DIR = Path("comparison_outputs")
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
SWEEP_COLOR_STOPS = [
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
]
SWEEP_DEFAULTS = {
    "population": {
        "total_agents": 100,
        "n_generations": 100,
        "n_rounds": 50,
        "fixed_drift": 0.02,
        "fixed_volatility": 0.10,
        "population_values": "20:80,30:70,40:60,50:50,60:40,70:30,80:20",
        "drift_values": "-0.05,-0.01,0.00,0.01,0.05,0.10,0.20",
        "volatility_values": "0.00,0.01,0.05,0.10,0.20,0.50",
    },
    "drift": {
        "total_agents": 100,
        "n_generations": 100,
        "n_rounds": 50,
        "fixed_drift": 0.02,
        "fixed_volatility": 0.10,
        "population_values": "20:80,30:70,40:60,50:50,60:40,70:30,80:20",
        "drift_values": "-0.05,-0.01,0.00,0.01,0.05,0.10,0.20",
        "volatility_values": "0.00,0.01,0.05,0.10,0.20,0.50",
    },
    "volatility": {
        "total_agents": 100,
        "n_generations": 100,
        "n_rounds": 50,
        "fixed_drift": 0.02,
        "fixed_volatility": 0.10,
        "population_values": "20:80,30:70,40:60,50:50,60:40,70:30,80:20",
        "drift_values": "-0.05,-0.01,0.00,0.01,0.05,0.10,0.20",
        "volatility_values": "0.00,0.01,0.05,0.10,0.20,0.50",
    },
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
    plot.getAxis("left").enableAutoSIPrefix(False)
    plot.getAxis("bottom").enableAutoSIPrefix(False)
    plot.getAxis("bottom").setStyle(tickTextOffset=6, autoExpandTextSpace=True)


def _resolve_plot_math_note(base_title: str) -> str:
    return PLOT_MATH_NOTES.get(base_title, "y = f(x)")


def _resolve_plot_brief_note(base_title: str, bottom_text: str | None = None, left_text: str | None = None) -> str:
    if base_title in PLOT_BRIEF_NOTES:
        return PLOT_BRIEF_NOTES[base_title]
    if bottom_text and left_text:
        return f"Shows how {left_text.lower()} changes with {bottom_text.lower()}."
    if left_text:
        return f"Shows how {left_text.lower()} evolves over the selected index."
    return "Shows the plotted series over the selected index."


def _set_plot_bottom_label(
    plot,
    x_label: str,
    legend_items=None,
    text_color: str = GRAPH_FOREGROUND,
):
    label_html = f"<span style='color: {text_color};'>{x_label}</span>"
    if legend_items:
        legend_html = "<br>".join(
            f"<span style='color: rgb({color[0]}, {color[1]}, {color[2]});'>&#9632; {name}</span>"
            for name, color in legend_items
        )
        label_html += f"<br><span style='font-size: 9pt;'>{legend_html}</span>"
    plot.setLabel("bottom", label_html)
    plot.getAxis("bottom").setHeight(54 if legend_items else 34)


def _format_run_label_html(legend_items, text_color: str = GRAPH_FOREGROUND):
    if not legend_items:
        return f"<span style='color: {text_color};'>No runs selected.</span>"
    return "<br>".join(
        f"<span style='color: rgb({color[0]}, {color[1]}, {color[2]});'>&#9632; {name}</span>"
        for name, color in legend_items
    )


def _pair_trade_links_from_agent_round(agent_round_df: pd.DataFrame) -> pd.DataFrame:
    if agent_round_df.empty:
        return pd.DataFrame()

    required_cols = {"round_number", "agent_id", "action", "executed_qty", "executed_price_avg"}
    if not required_cols.issubset(agent_round_df.columns):
        return pd.DataFrame()

    records = []
    for round_number, round_df in agent_round_df.groupby("round_number", sort=True):
        buyers = [
            {
                "agent_id": int(row["agent_id"]),
                "remaining_qty": float(row["executed_qty"]),
                "price": float(row["executed_price_avg"]) if pd.notna(row["executed_price_avg"]) else float("nan"),
            }
            for _, row in round_df.iterrows()
            if row["action"] == "buy" and pd.notna(row["executed_qty"]) and float(row["executed_qty"]) > 0
        ]
        sellers = [
            {
                "agent_id": int(row["agent_id"]),
                "remaining_qty": float(row["executed_qty"]),
                "price": float(row["executed_price_avg"]) if pd.notna(row["executed_price_avg"]) else float("nan"),
            }
            for _, row in round_df.iterrows()
            if row["action"] == "sell" and pd.notna(row["executed_qty"]) and float(row["executed_qty"]) > 0
        ]

        buyer_idx = 0
        seller_idx = 0
        trade_id = 1
        while buyer_idx < len(buyers) and seller_idx < len(sellers):
            buyer = buyers[buyer_idx]
            seller = sellers[seller_idx]
            matched_qty = min(buyer["remaining_qty"], seller["remaining_qty"])
            if matched_qty <= 1e-12:
                break

            if not np.isnan(buyer["price"]) and not np.isnan(seller["price"]):
                trade_price = (buyer["price"] + seller["price"]) / 2.0
            elif not np.isnan(buyer["price"]):
                trade_price = buyer["price"]
            elif not np.isnan(seller["price"]):
                trade_price = seller["price"]
            else:
                trade_price = 0.0

            records.append(
                {
                    "round_number": int(round_number),
                    "trade_id": int(trade_id),
                    "buyer_agent_id": int(buyer["agent_id"]),
                    "seller_agent_id": int(seller["agent_id"]),
                    "price": float(trade_price),
                    "quantity": float(matched_qty),
                    "notional": float(matched_qty * trade_price),
                }
            )
            trade_id += 1
            buyer["remaining_qty"] -= matched_qty
            seller["remaining_qty"] -= matched_qty
            if buyer["remaining_qty"] <= 1e-12:
                buyer_idx += 1
            if seller["remaining_qty"] <= 1e-12:
                seller_idx += 1

    return pd.DataFrame(records)


def _ensure_database_exists(db_path: str) -> bool:
    db_file = Path(db_path)
    database_created = not db_file.exists()
    create_database(str(db_file))
    return database_created


def _parse_float_list(raw_values: str) -> list[float]:
    values = []
    for part in raw_values.split(","):
        value = part.strip()
        if not value:
            continue
        values.append(float(value))
    if not values:
        raise ValueError("Please provide at least one numeric sweep value.")
    return values


def _parse_population_sweep(raw_values: str, total_agents: int) -> list[tuple[int, int]]:
    pairs = []
    for part in raw_values.split(","):
        value = part.strip()
        if not value:
            continue
        if ":" not in value:
            raise ValueError(
                "Population sweep values must use informed:zi pairs, for example 40:60,50:50."
            )
        informed_raw, zi_raw = [piece.strip() for piece in value.split(":", 1)]
        informed = int(informed_raw)
        zi = int(zi_raw)
        if informed + zi != total_agents:
            raise ValueError(
                f"Population sweep pair {informed}:{zi} does not sum to total agents ({total_agents})."
            )
        pairs.append((informed, zi))
    if not pairs:
        raise ValueError("Please provide at least one informed:zi population pair.")
    return pairs


def _interpolate_color(start_color, end_color, ratio: float):
    return tuple(
        int(round(start + (end - start) * ratio))
        for start, end in zip(start_color, end_color)
    )


def _sweep_colors(n_runs: int) -> list[tuple[int, int, int]]:
    if n_runs <= 0:
        return []
    if n_runs == 1:
        return [SWEEP_COLOR_STOPS[len(SWEEP_COLOR_STOPS) // 2]]

    colors = []
    last_index = len(SWEEP_COLOR_STOPS) - 1
    for idx in range(n_runs):
        scaled = (idx / (n_runs - 1)) * last_index
        low_index = int(np.floor(scaled))
        high_index = min(low_index + 1, last_index)
        ratio = scaled - low_index
        colors.append(
            _interpolate_color(
                SWEEP_COLOR_STOPS[low_index],
                SWEEP_COLOR_STOPS[high_index],
                ratio,
            )
        )
    return colors


def _build_sweep_title(sweep_name: str, settings: dict) -> str:
    total_agents = int(settings["total_agents"])
    fixed_drift = float(settings["fixed_drift"])
    fixed_volatility = float(settings["fixed_volatility"])
    default_informed = total_agents // 2
    default_zi = total_agents - default_informed

    if sweep_name == "population":
        return (
            f"Population Sweep (drift={fixed_drift:.2f}, vol={fixed_volatility:.2f}, "
            f"total={total_agents})"
        )
    if sweep_name == "drift":
        return (
            f"GBM Drift Sweep (vol={fixed_volatility:.2f}, "
            f"{default_informed} parametrised, {default_zi} ZI)"
        )
    return (
        f"GBM Volatility Sweep (drift={fixed_drift:.2f}, "
        f"{default_informed} parametrised, {default_zi} ZI)"
    )


def _build_sweep_run_args(sweep_name: str, settings: dict) -> tuple[str, list[tuple[str, dict]]]:
    total_agents = int(settings["total_agents"])
    n_generations = int(settings["n_generations"])
    n_rounds = int(settings["n_rounds"])
    fixed_drift = float(settings["fixed_drift"])
    fixed_volatility = float(settings["fixed_volatility"])
    title = _build_sweep_title(sweep_name, settings)

    run_args = []
    if sweep_name == "population":
        for n_informed, n_zi in _parse_population_sweep(settings["population_values"], total_agents):
            label = f"{n_informed} parametrised, {n_zi} ZI"
            run_args.append(
                (
                    label,
                    {
                        "n_parameterised_agents": n_informed,
                        "n_zi_agents": n_zi,
                        "n_generations": n_generations,
                        "n_rounds": n_rounds,
                        "GBM_drift": fixed_drift,
                        "GBM_volatility": fixed_volatility,
                    },
                )
            )
    elif sweep_name == "drift":
        default_informed = total_agents // 2
        default_zi = total_agents - default_informed
        for drift in _parse_float_list(settings["drift_values"]):
            label = f"Drift {drift:.2f}"
            run_args.append(
                (
                    label,
                    {
                        "n_parameterised_agents": default_informed,
                        "n_zi_agents": default_zi,
                        "n_generations": n_generations,
                        "n_rounds": n_rounds,
                        "GBM_drift": drift,
                        "GBM_volatility": fixed_volatility,
                    },
                )
            )
    elif sweep_name == "volatility":
        default_informed = total_agents // 2
        default_zi = total_agents - default_informed
        for volatility in _parse_float_list(settings["volatility_values"]):
            label = f"Vol {volatility:.2f}"
            run_args.append(
                (
                    label,
                    {
                        "n_parameterised_agents": default_informed,
                        "n_zi_agents": default_zi,
                        "n_generations": n_generations,
                        "n_rounds": n_rounds,
                        "GBM_drift": fixed_drift,
                        "GBM_volatility": volatility,
                    },
                )
            )
    else:
        raise ValueError(f"Unknown sweep type: {sweep_name}")

    if not run_args:
        raise ValueError("No runs were generated for the selected sweep.")
    return title, run_args


def _comparison_payload_to_runs(experiments_df: pd.DataFrame, comparison_df: pd.DataFrame) -> list[dict]:
    if experiments_df.empty or comparison_df.empty:
        return []

    labels_by_id = {
        str(row["experiment_id"]): f"{row['experiment_name']} | {str(row['experiment_id'])[:8]}"
        for _, row in experiments_df.iterrows()
    }
    runs = []
    for experiment_id, run_df in comparison_df.groupby("experiment_id", sort=False):
        run_copy = run_df.sort_values("generation_id").reset_index(drop=True).copy()
        run_copy["generation"] = run_copy["generation_id"].astype(int)
        runs.append(
            {
                "label": labels_by_id.get(str(experiment_id), str(experiment_id)),
                "experiment_id": str(experiment_id),
                "data": run_copy,
            }
        )
    return runs


class DatabaseLoaderWorker(QThread):
    """Background thread that loads experiment and generation data from a DuckDB file."""

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
        database_created = _ensure_database_exists(str(db_file))

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
            trade_execution_df = pd.DataFrame()
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


class ExperimentRunnerWorker(QThread):
    """Background thread that executes a full experiment run and emits progress events to the GUI."""

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


class ComparisonLoaderWorker(QThread):
    """Background thread that fetches generation-level metrics for multiple experiments to populate comparison charts."""

    loaded = Signal(dict)
    error = Signal(str)

    def __init__(self, db_path, experiment_ids):
        super().__init__()
        self.db_path = db_path
        self.experiment_ids = [str(experiment_id) for experiment_id in experiment_ids]

    def run(self):
        try:
            self.loaded.emit(self._load_payload())
        except Exception as exc:
            self.error.emit(str(exc))

    def _load_payload(self):
        db_file = Path(self.db_path)
        _ensure_database_exists(str(db_file))
        if not self.experiment_ids:
            return {
                "experiments": pd.DataFrame(),
                "comparison_metrics": pd.DataFrame(),
                "sql_objects": {},
            }

        con = duckdb.connect(str(db_file))
        try:
            experiment_ids_sql = ", ".join(f"'{experiment_id}'" for experiment_id in self.experiment_ids)

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
                    if "experiment_id" in column_name_set:
                        where_clauses.append(f"experiment_id IN ({experiment_ids_sql})")

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
                            f"SELECT COUNT(*) FROM ({base_query}) AS filtered_object"
                        ).fetchone()[0]
                    )
                    sql_objects[str(object_name)] = {
                        "display_name": _humanize_sql_object_name(str(object_name)),
                        "type": str(object_type),
                        "row_count": total_rows,
                        "preview_rows": min(total_rows, SQL_TAB_PREVIEW_ROWS),
                        "data": con.execute(
                            f"{base_query} LIMIT {SQL_TAB_PREVIEW_ROWS}"
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
                "comparison_metrics": comparison_df,
                "sql_objects": sql_objects,
            }
        finally:
            con.close()


class SweepComparisonWorker(QThread):
    """Background thread that runs persisted sweep experiments and streams live comparison updates."""

    progress = Signal(dict)
    completed = Signal(dict)
    error = Signal(str)

    def __init__(self, sweep_name: str, settings: dict):
        super().__init__()
        self.sweep_name = sweep_name
        self.settings = dict(settings)

    def run(self):
        try:
            sweep_title, run_args = _build_sweep_run_args(self.sweep_name, self.settings)
            total_runs = len(run_args)
            self.progress.emit(
                {
                    "event": "sweep_started",
                    "sweep_name": self.sweep_name,
                    "sweep_title": sweep_title,
                    "total_runs": total_runs,
                }
            )
            indexed_results = []
            for run_index, (label, overrides) in enumerate(run_args, start=1):
                run_overrides = dict(overrides)
                run_overrides["db_path"] = str(self.settings["db_path"])
                run_overrides["experiment_name"] = label
                run_overrides["experiment_type"] = f"Sweep: {self.sweep_name}"
                run_overrides["run_notes"] = sweep_title

                live_generation_df = pd.DataFrame()
                self.progress.emit(
                    {
                        "event": "sweep_run_started",
                        "sweep_name": self.sweep_name,
                        "sweep_title": sweep_title,
                        "run_index": run_index,
                        "total_runs": total_runs,
                        "run_label": label,
                    }
                )

                def sweep_progress_callback(payload):
                    nonlocal live_generation_df
                    payload = dict(payload)
                    payload["sweep_name"] = self.sweep_name
                    payload["sweep_title"] = sweep_title
                    payload["run_index"] = run_index
                    payload["total_runs"] = total_runs
                    payload["run_label"] = label
                    if payload.get("event") == "generation_completed":
                        metrics = payload.get("generation_metrics", {})
                        if metrics:
                            live_generation_df = pd.concat(
                                [live_generation_df, pd.DataFrame([metrics])],
                                ignore_index=True,
                            )
                            live_generation_df = (
                                live_generation_df.drop_duplicates(
                                    subset=["generation_id"],
                                    keep="last",
                                )
                                .sort_values("generation_id")
                                .reset_index(drop=True)
                            )
                            payload["live_generation_df"] = live_generation_df.copy()
                    self.progress.emit(payload)

                result = run_experiment(
                    config_overrides=run_overrides,
                    progress_callback=sweep_progress_callback,
                    run_analysis=False,
                )
                run_df = result["generation_counts_df"].reset_index(drop=True).copy()
                if "generation" in run_df.columns:
                    run_df["generation"] = run_df["generation"].astype(int)
                indexed_results.append(
                    {
                        "label": label,
                        "experiment_id": result.get("experiment_id"),
                        "data": run_df,
                    }
                )
                self.progress.emit(
                    {
                        "event": "sweep_run_completed",
                        "sweep_name": self.sweep_name,
                        "run_label": label,
                        "run_index": run_index,
                        "completed_runs": run_index,
                        "total_runs": total_runs,
                        "experiment_id": result.get("experiment_id"),
                        "data": run_df.copy(),
                    }
                )

            self.completed.emit(
                {
                    "sweep_name": self.sweep_name,
                    "sweep_title": sweep_title,
                    "runs": indexed_results,
                    "settings": self.settings,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))


class CommandCenter(QMainWindow):
    """Main application window hosting the experiment runner, live charts, database browser, and comparison tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent-Based Market Simulation")
        self.resize(1600, 980)

        self._theme_mode = "light"
        self._theme = APP_THEMES[self._theme_mode]
        self._current_experiment_id = None
        self._current_generation_id = None
        self._microstructure_round = None
        self._suppress_selection_signals = False
        self.worker = None
        self.run_worker = None
        self.compare_worker = None
        self.sweep_compare_worker = None
        self.live_generations_df = pd.DataFrame()
        self.live_strategy_generation_df = pd.DataFrame()
        self.live_strategy_round_df = pd.DataFrame()
        self._pending_generation_id = None
        self._last_payload = None
        self._comparison_payload = None
        self._comparison_sweep_payload = None
        self._comparison_selected_ids = []
        self._comparison_legend_items = []
        self._comparison_sweep_legend_items = []
        self.smoothing_window = 1
        self._pending_smoothing_window = 1
        self._plot_theme_specs = {}
        self._plot_canvases = []
        self._comparison_zero_lines = {}
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
        self.microstructure_tab = QWidget()
        self._build_microstructure_tab()
        self.comparison_tab = QWidget()
        self._build_comparison_tab()
        self.sql_tab = QTabWidget()
        self._build_sql_tab()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.strategy_performance_tab, "Strategy Performance")
        self.tabs.addTab(self.agent_performance_tab, "Agent Performance")
        self.tabs.addTab(self.microstructure_tab, "Market Microstructure")
        self.tabs.addTab(self.comparison_tab, "Run Comparison")
        self.tabs.addTab(self.sql_tab, "SQL Data")

        self.combo_experiment.currentIndexChanged.connect(self._on_experiment_changed)
        self.combo_generation.currentIndexChanged.connect(self._on_generation_changed)
        self.generation_slider.valueChanged.connect(self._on_generation_slider_changed)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        self.checkbox_show_parameterised.toggled.connect(self._refresh_plots_only)
        self.checkbox_show_zi.toggled.connect(self._refresh_plots_only)
        self.checkbox_dark_mode.toggled.connect(self._on_theme_toggled)

        self._apply_theme()
        self.refresh_data()

    def _register_plot_canvas(self, canvas):
        self._plot_canvases.append(canvas)

    def _register_plot_theme(self, plot, *, bottom_text=None, bottom_legend=None, left_text=None, title_text=None):
        self._plot_theme_specs[id(plot)] = {
            "plot": plot,
            "bottom_text": bottom_text,
            "bottom_legend": bottom_legend,
            "left_text": left_text,
            "title_text": title_text,
        }

    def _create_plot_explanation_label(self, titles):
        lines = []
        for title in titles:
            math_note = _resolve_plot_math_note(title)
            brief_note = _resolve_plot_brief_note(title)
            lines.append(
                f"<b>{title}</b><br>"
                f"<span style='font-size: 9pt;'>{math_note}</span><br>"
                f"<span style='font-size: 9pt;'>{brief_note}</span>"
            )
        label = QLabel("<br><br>".join(lines))
        label.setObjectName("plotExplanation")
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        return label

    def _build_stylesheet(self):
        theme = self._theme
        return f"""
        QWidget {{
            background-color: {theme["window_bg"]};
            color: {theme["text"]};
        }}
        QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget, QTabWidget::pane {{
            background-color: {theme["window_bg"]};
        }}
        QGroupBox {{
            background-color: {theme["surface_bg"]};
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }}
        QLineEdit, QPlainTextEdit, QListWidget, QComboBox, QTableWidget, QTabBar::tab {{
            background-color: {theme["input_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-radius: 6px;
            padding: 6px;
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QTabBar::tab:selected {{
            background-color: {theme["surface_bg"]};
        }}
        QHeaderView::section {{
            background-color: {theme["surface_alt_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            padding: 6px;
        }}
        QPushButton {{
            background-color: {theme["surface_alt_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            border-color: {theme["selection_bg"]};
        }}
        QPushButton:disabled {{
            color: {theme["muted_text"]};
        }}
        QPushButton#refreshButton {{
            background-color: {theme["refresh_button"]};
            color: #ffffff;
            border: none;
        }}
        QPushButton#startRunButton {{
            background-color: {theme["start_button"]};
            color: #ffffff;
            border: none;
        }}
        QPushButton#stopRunButton {{
            background-color: {theme["stop_button"]};
            color: #ffffff;
            border: none;
        }}
        QPushButton#compareRunsButton {{
            background-color: {theme["compare_button"]};
            color: #ffffff;
            border: none;
        }}
        QCheckBox, QLabel {{
            color: {theme["text"]};
        }}
        QLabel#plotExplanation {{
            background-color: {theme["surface_alt_bg"]};
            border: 1px solid {theme["border"]};
            border-radius: 6px;
            padding: 8px 10px;
            margin-top: 6px;
            color: {theme["muted_text"]};
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: {theme["surface_alt_bg"]};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {theme["selection_bg"]};
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}
        QAbstractItemView {{
            selection-background-color: {theme["selection_bg"]};
            selection-color: {theme["selection_text"]};
        }}
        """

    def _apply_theme_to_plot(self, plot):
        theme = self._theme
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(theme["plot_foreground"]))
            axis.setTextPen(pg.mkPen(theme["plot_foreground"]))

        spec = self._plot_theme_specs.get(id(plot))
        if spec:
            if spec["left_text"]:
                plot.setLabel("left", spec["left_text"], color=theme["plot_foreground"])
            if spec["bottom_text"]:
                _set_plot_bottom_label(
                    plot,
                    spec["bottom_text"],
                    spec["bottom_legend"],
                    text_color=theme["plot_foreground"],
                )

    def _apply_theme(self):
        self._theme = APP_THEMES[self._theme_mode]
        pg.setConfigOption("background", self._theme["plot_background"])
        pg.setConfigOption("foreground", self._theme["plot_foreground"])
        self.setStyleSheet(self._build_stylesheet())

        if hasattr(self, "plot_market"):
            market_spec = self._plot_theme_specs.get(id(self.plot_market))
            if market_spec and market_spec["bottom_legend"]:
                market_spec["bottom_legend"][-1] = ("Mid Price", self._theme["plot_mid_price"])

        for canvas in self._plot_canvases:
            canvas.setBackground(self._theme["plot_background"])
        for spec in self._plot_theme_specs.values():
            self._apply_theme_to_plot(spec["plot"])

        if hasattr(self, "line_mid_price"):
            self.line_mid_price.setPen(pg.mkPen(self._theme["plot_mid_price"], width=3))
        for zero_line in self._comparison_zero_lines.values():
            zero_line.setPen(pg.mkPen(self._theme["plot_reference"], width=1, style=Qt.DashLine))

        if hasattr(self, "comparison_param_runs_label"):
            labels_html = _format_run_label_html(
                self._comparison_legend_items,
                text_color=self._theme["plot_foreground"],
            )
            self.comparison_param_runs_label.setText(labels_html)
            self.comparison_wealth_runs_label.setText(labels_html)

        if hasattr(self, "checkbox_dark_mode"):
            self.checkbox_dark_mode.blockSignals(True)
            self.checkbox_dark_mode.setChecked(self._theme_mode == "dark")
            self.checkbox_dark_mode.blockSignals(False)

        if hasattr(self, "plot_params"):
            self._update_all_plot_titles()
            self._refresh_plots_only()

    def _on_theme_toggled(self, checked):
        self._theme_mode = "dark" if checked else "light"
        self._apply_theme()

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
        self.checkbox_dark_mode = QCheckBox("Dark mode")
        self.checkbox_dark_mode.setChecked(False)
        data_form.addRow("DuckDB File:", self.input_db_path)
        data_form.addRow("Experiment:", self.combo_experiment)
        data_form.addRow("Generation:", self.combo_generation)
        data_form.addRow("Generation Scroll:", self.generation_slider)
        data_form.addRow("", self.generation_slider_label)
        data_form.addRow("Graph Smoothing:", self.smoothing_slider)
        data_form.addRow("", self.smoothing_slider_label)
        data_form.addRow("Series Visibility:", self.checkbox_show_parameterised)
        data_form.addRow("", self.checkbox_show_zi)
        data_form.addRow("Appearance:", self.checkbox_dark_mode)
        data_group.setLayout(data_form)
        left_layout.addWidget(data_group)

        self.btn_refresh = QPushButton("Refresh Database")
        self.btn_refresh.setObjectName("refreshButton")
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
        self.btn_start_run.setObjectName("startRunButton")
        self.btn_start_run.clicked.connect(self.start_new_run)
        left_layout.addWidget(self.btn_start_run)

        self.btn_stop_run = QPushButton("Stop Current Run")
        self.btn_stop_run.setObjectName("stopRunButton")
        self.btn_stop_run.clicked.connect(self.stop_run)
        self.btn_stop_run.setEnabled(False)
        left_layout.addWidget(self.btn_stop_run)

        self.run_status_label = QLabel("No run started.")
        self.run_status_label.setWordWrap(True)
        left_layout.addWidget(self.run_status_label)

        comparison_group = QGroupBox("Compare Multiple Runs")
        comparison_layout = QVBoxLayout()
        comparison_help = QLabel(
            "Select two or more experiments to overlay their generation-level parameter and wealth trends."
        )
        comparison_help.setWordWrap(True)
        self.comparison_experiment_list = QListWidget()
        self.comparison_experiment_list.setSelectionMode(QListWidget.MultiSelection)
        self.btn_compare_runs = QPushButton("Compare Selected Runs")
        self.btn_compare_runs.setObjectName("compareRunsButton")
        self.btn_compare_runs.clicked.connect(self.load_comparison_data)
        self.btn_clear_comparison = QPushButton("Clear Comparison")
        self.btn_clear_comparison.clicked.connect(self.clear_comparison)
        self.comparison_status_label = QLabel("Select experiments and click Compare Selected Runs.")
        self.comparison_status_label.setWordWrap(True)
        sweep_help = QLabel(
            "Run sweep comparisons as normal persisted experiments. The comparison plots update live as each generation finishes and PNG snapshots are saved to comparison_outputs."
        )
        sweep_help.setWordWrap(True)
        sweep_form = QFormLayout()
        self.combo_sweep_type = QComboBox()
        self.combo_sweep_type.addItems(["population", "drift", "volatility"])
        self.combo_sweep_type.currentTextChanged.connect(self._apply_sweep_defaults)
        self.sweep_total_agents_input = QLineEdit()
        self.sweep_generations_input = QLineEdit()
        self.sweep_rounds_input = QLineEdit()
        self.sweep_fixed_drift_input = QLineEdit()
        self.sweep_fixed_volatility_input = QLineEdit()
        self.sweep_population_values_input = QLineEdit()
        self.sweep_drift_values_input = QLineEdit()
        self.sweep_volatility_values_input = QLineEdit()
        sweep_form.addRow("Sweep Type:", self.combo_sweep_type)
        sweep_form.addRow("Total Agents:", self.sweep_total_agents_input)
        sweep_form.addRow("Generations:", self.sweep_generations_input)
        sweep_form.addRow("Rounds:", self.sweep_rounds_input)
        sweep_form.addRow("Fixed Drift:", self.sweep_fixed_drift_input)
        sweep_form.addRow("Fixed Volatility:", self.sweep_fixed_volatility_input)
        sweep_form.addRow("Population Pairs:", self.sweep_population_values_input)
        sweep_form.addRow("Drift Values:", self.sweep_drift_values_input)
        sweep_form.addRow("Volatility Values:", self.sweep_volatility_values_input)
        self.btn_run_sweep_comparison = QPushButton("Run Sweep Comparison")
        self.btn_run_sweep_comparison.setObjectName("compareRunsButton")
        self.btn_run_sweep_comparison.clicked.connect(self.run_sweep_comparison)
        self.btn_clear_sweep_comparison = QPushButton("Clear Sweep Comparison")
        self.btn_clear_sweep_comparison.clicked.connect(self.clear_sweep_comparison)
        self.sweep_status_label = QLabel("Select a sweep and click Run Sweep Comparison.")
        self.sweep_status_label.setWordWrap(True)
        comparison_layout.addWidget(comparison_help)
        comparison_layout.addWidget(self.comparison_experiment_list)
        comparison_layout.addWidget(self.btn_compare_runs)
        comparison_layout.addWidget(self.btn_clear_comparison)
        comparison_layout.addWidget(self.comparison_status_label)
        comparison_layout.addWidget(sweep_help)
        comparison_layout.addLayout(sweep_form)
        comparison_layout.addWidget(self.btn_run_sweep_comparison)
        comparison_layout.addWidget(self.btn_clear_sweep_comparison)
        comparison_layout.addWidget(self.sweep_status_label)
        comparison_group.setLayout(comparison_layout)
        left_layout.addWidget(comparison_group)
        self._apply_sweep_defaults(self.combo_sweep_type.currentText())
        left_layout.addStretch()

    def _build_dashboard_tab(self):
        layout = QVBoxLayout(self.dashboard_tab)
        dashboard_scroll = QScrollArea()
        dashboard_scroll.setWidgetResizable(True)
        dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(dashboard_scroll)

        dashboard_content = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_content)
        self.graph_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.graph_area)
        self.graph_area.setMinimumHeight(1800)
        dashboard_layout.addWidget(self.graph_area)
        dashboard_layout.addWidget(
            self._create_plot_explanation_label(
                [
                    "Mean Strategy Parameters Across Generations",
                    "Mean Wealth by Strategy",
                    "Mean Info_Param by Strategy",
                    "Market Summary by Round",
                    "Average Profit per Round: ZI vs Parameterised",
                    "Average Agent Volume Share per Round",
                ]
            )
        )
        dashboard_scroll.setWidget(dashboard_content)

        self.plot_params_title = "Mean Strategy Parameters Across Generations"
        self.plot_wealth_title = "Mean Wealth by Strategy"
        self.plot_info_param_title = "Mean Info_Param by Strategy"
        self.plot_market_title = "Market Summary by Round"
        self.plot_profit_title = "Average Profit per Round: ZI vs Parameterised"
        self.plot_volume_share_title = "Average Agent Volume Share per Round"

        self.plot_params = self.graph_area.addPlot(title=self._format_plot_title(self.plot_params_title))
        _style_plot(self.plot_params)
        self._register_plot_theme(
            self.plot_params,
            bottom_text="Generation",
            bottom_legend=[
                ("Qty Aggression", (255, 140, 0)),
                ("Signal Aggression", (70, 130, 180)),
            ],
            left_text="Mean Parameter Value",
            title_text=self.plot_params_title,
        )
        self.line_qty = self.plot_params.plot(pen=pg.mkPen((255, 140, 0), width=3), name="Qty Aggression")
        self.line_signal = self.plot_params.plot(pen=pg.mkPen((70, 130, 180), width=3), name="Signal Aggression")

        self.graph_area.nextRow()

        self.plot_wealth = self.graph_area.addPlot(title=self._format_plot_title(self.plot_wealth_title))
        _style_plot(self.plot_wealth)
        self._register_plot_theme(
            self.plot_wealth,
            bottom_text="Generation",
            bottom_legend=[
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
            left_text="Mean Wealth",
            title_text=self.plot_wealth_title,
        )
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
        self._register_plot_theme(
            self.plot_info_param,
            bottom_text="Generation",
            bottom_legend=[
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
            left_text="Mean Info_Param",
            title_text=self.plot_info_param_title,
        )
        self.line_info_param_informed = self.plot_info_param.plot(
            pen=pg.mkPen((30, 144, 255), width=3), name="Parameterised Informed"
        )
        self.line_info_param_zi = self.plot_info_param.plot(
            pen=pg.mkPen((205, 92, 92), width=3), name="ZI"
        )

        self.graph_area.nextRow()

        self.plot_market = self.graph_area.addPlot(title=self._format_plot_title(self.plot_market_title))
        _style_plot(self.plot_market)
        self._register_plot_theme(
            self.plot_market,
            bottom_text="Round",
            bottom_legend=[
                ("Max Bid", (30, 144, 255)),
                ("Max Sell", (220, 20, 60)),
                ("Min Bid", (135, 206, 250)),
                ("Min Sell", (250, 128, 114)),
                ("Bid Price Q2", (65, 105, 225)),
                ("Ask Price Q3", (178, 34, 34)),
                ("Fundamental", (128, 128, 128)),
                ("Mid Price", self._theme["plot_mid_price"]),
            ],
            left_text="Price",
            title_text=self.plot_market_title,
        )
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
            pen=pg.mkPen(self._theme["plot_mid_price"], width=3), name="Mid Price"
        )

        self.graph_area.nextRow()

        self.plot_profit = self.graph_area.addPlot(title=self._format_plot_title(self.plot_profit_title))
        _style_plot(self.plot_profit)
        self._register_plot_theme(
            self.plot_profit,
            bottom_text="Round",
            bottom_legend=[
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
            left_text="Average Profit",
            title_text=self.plot_profit_title,
        )
        self.line_profit_informed = self.plot_profit.plot(
            pen=pg.mkPen((30, 144, 255), width=3), name="Parameterised Informed"
        )
        self.line_profit_zi = self.plot_profit.plot(
            pen=pg.mkPen((205, 92, 92), width=3), name="ZI"
        )

        self.graph_area.nextRow()

        self.plot_volume_share = self.graph_area.addPlot(title=self._format_plot_title(self.plot_volume_share_title))
        _style_plot(self.plot_volume_share)
        self._register_plot_theme(
            self.plot_volume_share,
            bottom_text="Round",
            bottom_legend=[
                ("Parameterised Informed", (30, 144, 255)),
                ("ZI", (205, 92, 92)),
            ],
            left_text="Average Volume Share",
            title_text=self.plot_volume_share_title,
        )
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

    def _build_comparison_tab(self):
        layout = QVBoxLayout(self.comparison_tab)
        comparison_scroll = QScrollArea()
        comparison_scroll.setWidgetResizable(True)
        comparison_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(comparison_scroll)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        summary_group = QGroupBox("Selected Run Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.comparison_summary_label = QLabel(
            "No comparison loaded. Select experiments from the left panel."
        )
        self.comparison_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.comparison_summary_label)
        content_layout.addWidget(summary_group)

        parameter_group = QGroupBox("Parameter Comparison Across Generations")
        parameter_layout = QVBoxLayout(parameter_group)
        self.comparison_param_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.comparison_param_area)
        self.comparison_param_area.setMinimumHeight(1300)
        parameter_layout.addWidget(self.comparison_param_area)
        self.comparison_param_runs_label = QLabel("No runs selected.")
        self.comparison_param_runs_label.setWordWrap(True)
        parameter_layout.addWidget(self.comparison_param_runs_label)
        parameter_layout.addWidget(
            self._create_plot_explanation_label(
                [title for _, title in COMPARISON_PARAM_SPECS]
            )
        )
        content_layout.addWidget(parameter_group)

        wealth_group = QGroupBox("Wealth Difference Across Generations")
        wealth_layout = QVBoxLayout(wealth_group)
        self.comparison_wealth_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.comparison_wealth_area)
        self.comparison_wealth_area.setMinimumHeight(450)
        wealth_layout.addWidget(self.comparison_wealth_area)
        self.comparison_wealth_runs_label = QLabel("No runs selected.")
        self.comparison_wealth_runs_label.setWordWrap(True)
        wealth_layout.addWidget(self.comparison_wealth_runs_label)
        wealth_layout.addWidget(
            self._create_plot_explanation_label([COMPARISON_WEALTH_DIFF_SPEC[1]])
        )
        content_layout.addWidget(wealth_group)

        sweep_summary_group = QGroupBox("Sweep Run Summary")
        sweep_summary_layout = QVBoxLayout(sweep_summary_group)
        self.sweep_summary_label = QLabel(
            "No sweep comparison loaded. Configure a sweep from the left panel."
        )
        self.sweep_summary_label.setWordWrap(True)
        sweep_summary_layout.addWidget(self.sweep_summary_label)
        content_layout.addWidget(sweep_summary_group)

        sweep_parameter_group = QGroupBox("Sweep Parameters Across Generations")
        sweep_parameter_layout = QVBoxLayout(sweep_parameter_group)
        self.sweep_param_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.sweep_param_area)
        self.sweep_param_area.setMinimumHeight(1000)
        sweep_parameter_layout.addWidget(self.sweep_param_area)
        self.sweep_param_runs_label = QLabel("No sweep runs selected.")
        self.sweep_param_runs_label.setWordWrap(True)
        sweep_parameter_layout.addWidget(self.sweep_param_runs_label)
        sweep_parameter_layout.addWidget(
            self._create_plot_explanation_label(
                [title for _, title in SWEEP_PARAM_SUBPLOTS]
            )
        )
        content_layout.addWidget(sweep_parameter_group)

        sweep_diversity_group = QGroupBox("Sweep Diversity Across Generations")
        sweep_diversity_layout = QVBoxLayout(sweep_diversity_group)
        self.sweep_diversity_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.sweep_diversity_area)
        self.sweep_diversity_area.setMinimumHeight(1000)
        sweep_diversity_layout.addWidget(self.sweep_diversity_area)
        self.sweep_diversity_runs_label = QLabel("No sweep runs selected.")
        self.sweep_diversity_runs_label.setWordWrap(True)
        sweep_diversity_layout.addWidget(self.sweep_diversity_runs_label)
        sweep_diversity_layout.addWidget(
            self._create_plot_explanation_label(
                [title for _, title in SWEEP_STD_SUBPLOTS]
            )
        )
        content_layout.addWidget(sweep_diversity_group)

        sweep_wealth_group = QGroupBox("Sweep Wealth Difference Across Generations")
        sweep_wealth_layout = QVBoxLayout(sweep_wealth_group)
        self.sweep_wealth_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.sweep_wealth_area)
        self.sweep_wealth_area.setMinimumHeight(450)
        sweep_wealth_layout.addWidget(self.sweep_wealth_area)
        self.sweep_wealth_runs_label = QLabel("No sweep runs selected.")
        self.sweep_wealth_runs_label.setWordWrap(True)
        sweep_wealth_layout.addWidget(self.sweep_wealth_runs_label)
        sweep_wealth_layout.addWidget(
            self._create_plot_explanation_label([COMPARISON_WEALTH_DIFF_SPEC[1]])
        )
        content_layout.addWidget(sweep_wealth_group)

        comparison_scroll.setWidget(content)

        self.comparison_param_plots = {}
        self.comparison_wealth_plots = {}
        self.comparison_param_curves = {}
        self.sweep_param_plots = {}
        self.sweep_diversity_plots = {}
        self.sweep_wealth_plots = {}
        self.comparison_plot_titles = {}
        self.sweep_plot_titles = {}

        for idx, (metric_key, title) in enumerate(COMPARISON_PARAM_SPECS):
            plot_kwargs = {"title": self._format_plot_title(title)}
            if metric_key == "mean_info_param_parameterised_informed":
                plot_kwargs.update({"row": idx // 2, "col": 0, "colspan": 2})
            plot = self.comparison_param_area.addPlot(**plot_kwargs)
            _style_plot(plot)
            self._register_plot_theme(plot, bottom_text="Generation", left_text="Value", title_text=title)
            self.comparison_param_plots[metric_key] = plot
            self.comparison_param_curves[metric_key] = {}
            self.comparison_plot_titles[metric_key] = title
            if idx % 2 == 1 and metric_key != "mean_info_param_parameterised_informed":
                self.comparison_param_area.nextRow()
        self.comparison_param_area.ci.layout.setColumnStretchFactor(0, 1)
        self.comparison_param_area.ci.layout.setColumnStretchFactor(1, 1)

        wealth_metric_key, wealth_title = COMPARISON_WEALTH_DIFF_SPEC
        wealth_plot = self.comparison_wealth_area.addPlot(
            row=0,
            col=0,
            colspan=2,
            title=self._format_plot_title(wealth_title),
        )
        _style_plot(wealth_plot)
        self._register_plot_theme(
            wealth_plot,
            bottom_text="Generation",
            left_text="Wealth Difference",
            title_text=wealth_title,
        )
        self._comparison_zero_lines[wealth_metric_key] = wealth_plot.addLine(
            y=0,
            pen=pg.mkPen(self._theme["plot_reference"], width=1, style=Qt.DashLine),
        )
        self.comparison_wealth_plots[wealth_metric_key] = wealth_plot
        self.comparison_plot_titles[wealth_metric_key] = wealth_title
        self.comparison_wealth_area.ci.layout.setColumnStretchFactor(0, 1)
        self.comparison_wealth_area.ci.layout.setColumnStretchFactor(1, 1)

        for idx, (metric_key, title) in enumerate(SWEEP_PARAM_SUBPLOTS):
            plot_kwargs = {"title": self._format_plot_title(title)}
            if metric_key == "mean_info_param_parameterised_informed":
                plot_kwargs.update({"row": idx // 2, "col": 0, "colspan": 2})
            plot = self.sweep_param_area.addPlot(**plot_kwargs)
            _style_plot(plot)
            self._register_plot_theme(plot, bottom_text="Generation", left_text="Mean Value", title_text=title)
            self.sweep_param_plots[metric_key] = plot
            self.sweep_plot_titles[metric_key] = title
            if idx % 2 == 1 and metric_key != "mean_info_param_parameterised_informed":
                self.sweep_param_area.nextRow()
        self.sweep_param_area.ci.layout.setColumnStretchFactor(0, 1)
        self.sweep_param_area.ci.layout.setColumnStretchFactor(1, 1)

        for idx, (metric_key, title) in enumerate(SWEEP_STD_SUBPLOTS):
            plot_kwargs = {"title": self._format_plot_title(title)}
            if metric_key == "std_info_param_parameterised_informed":
                plot_kwargs.update({"row": idx // 2, "col": 0, "colspan": 2})
            plot = self.sweep_diversity_area.addPlot(**plot_kwargs)
            _style_plot(plot)
            self._register_plot_theme(plot, bottom_text="Generation", left_text="Std Dev", title_text=title)
            self.sweep_diversity_plots[metric_key] = plot
            self.sweep_plot_titles[metric_key] = title
            if idx % 2 == 1 and metric_key != "std_info_param_parameterised_informed":
                self.sweep_diversity_area.nextRow()
        self.sweep_diversity_area.ci.layout.setColumnStretchFactor(0, 1)
        self.sweep_diversity_area.ci.layout.setColumnStretchFactor(1, 1)

        sweep_wealth_metric_key, sweep_wealth_title = COMPARISON_WEALTH_DIFF_SPEC
        sweep_wealth_plot = self.sweep_wealth_area.addPlot(
            row=0,
            col=0,
            colspan=2,
            title=self._format_plot_title(sweep_wealth_title)
        )
        _style_plot(sweep_wealth_plot)
        self._register_plot_theme(
            sweep_wealth_plot,
            bottom_text="Generation",
            left_text="Wealth Difference",
            title_text=sweep_wealth_title,
        )
        sweep_wealth_plot.addLine(
            y=0,
            pen=pg.mkPen(self._theme["plot_reference"], width=1, style=Qt.DashLine),
        )
        self.sweep_wealth_plots[sweep_wealth_metric_key] = sweep_wealth_plot
        self.sweep_plot_titles[sweep_wealth_metric_key] = sweep_wealth_title
        self.sweep_wealth_area.ci.layout.setColumnStretchFactor(0, 1)
        self.sweep_wealth_area.ci.layout.setColumnStretchFactor(1, 1)

    def _build_microstructure_tab(self):
        layout = QVBoxLayout(self.microstructure_tab)
        microstructure_scroll = QScrollArea()
        microstructure_scroll.setWidgetResizable(True)
        microstructure_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(microstructure_scroll)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        summary_group = QGroupBox("Microstructure Overview")
        summary_layout = QVBoxLayout(summary_group)
        self.microstructure_summary_label = QLabel(
            "Load an experiment and generation to inspect market microstructure."
        )
        self.microstructure_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.microstructure_summary_label)
        self.micro_round_slider = QSlider(Qt.Horizontal)
        self.micro_round_slider.setEnabled(False)
        self.micro_round_slider.setMinimum(0)
        self.micro_round_slider.setMaximum(0)
        self.micro_round_slider.setValue(0)
        self.micro_round_slider.valueChanged.connect(self._on_micro_round_changed)
        self.micro_round_slider_label = QLabel("Trade network round: unavailable")
        summary_layout.addWidget(self.micro_round_slider)
        summary_layout.addWidget(self.micro_round_slider_label)
        content_layout.addWidget(summary_group)

        self.microstructure_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.microstructure_area)
        self.microstructure_area.setMinimumHeight(1800)
        content_layout.addWidget(self.microstructure_area)
        content_layout.addWidget(
            self._create_plot_explanation_label(
                [
                    "Limit Order Book Snapshot",
                    "Candlestick View",
                    "Trade Network",
                    "Order Imbalance and Spread",
                    "Participation and Volume",
                ]
            )
        )
        microstructure_scroll.setWidget(content)

        self.micro_lob_title = "Limit Order Book Snapshot"
        self.micro_candle_title = "Candlestick View"
        self.micro_network_title = "Trade Network"
        self.micro_pressure_title = "Order Imbalance and Spread"
        self.micro_activity_title = "Participation and Volume"

        self.micro_lob_plot = self.microstructure_area.addPlot(
            title=self._format_plot_title(self.micro_lob_title)
        )
        _style_plot(self.micro_lob_plot)
        self._register_plot_theme(
            self.micro_lob_plot,
            bottom_text="Limit Price",
            left_text="Order Qty",
            title_text=self.micro_lob_title,
        )

        self.micro_candle_plot = self.microstructure_area.addPlot(
            title=self._format_plot_title(self.micro_candle_title)
        )
        _style_plot(self.micro_candle_plot)
        self._register_plot_theme(
            self.micro_candle_plot,
            bottom_text="Round",
            left_text="Price",
            title_text=self.micro_candle_title,
        )

        self.microstructure_area.nextRow()

        self.micro_network_plot = self.microstructure_area.addPlot(
            title=self._format_plot_title(self.micro_network_title)
        )
        _style_plot(self.micro_network_plot)
        self._register_plot_theme(
            self.micro_network_plot,
            bottom_text="X",
            left_text="Y",
            title_text=self.micro_network_title,
        )
        self.micro_network_plot.hideAxis("left")
        self.micro_network_plot.hideAxis("bottom")

        self.micro_pressure_plot = self.microstructure_area.addPlot(
            title=self._format_plot_title(self.micro_pressure_title)
        )
        _style_plot(self.micro_pressure_plot)
        self._register_plot_theme(
            self.micro_pressure_plot,
            bottom_text="Round",
            left_text="Value",
            title_text=self.micro_pressure_title,
        )

        self.microstructure_area.nextRow()

        self.micro_activity_plot = self.microstructure_area.addPlot(
            row=2,
            col=0,
            colspan=2,
            title=self._format_plot_title(self.micro_activity_title)
        )
        _style_plot(self.micro_activity_plot)
        self._register_plot_theme(
            self.micro_activity_plot,
            bottom_text="Round",
            left_text="Value",
            title_text=self.micro_activity_title,
        )
        self.microstructure_area.ci.layout.setColumnStretchFactor(0, 1)
        self.microstructure_area.ci.layout.setColumnStretchFactor(1, 1)

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
        self._register_plot_canvas(self.performance_generation_area)
        self.performance_generation_area.setMinimumHeight(1500)
        generation_layout.addWidget(self.performance_generation_area)
        content_layout.addWidget(generation_group)

        round_group = QGroupBox("Strategy Performance Across Rounds")
        round_layout = QVBoxLayout(round_group)
        self.performance_round_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.performance_round_area)
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
        generation_layout.addWidget(
            self._create_plot_explanation_label(
                [title for _, title in generation_metrics]
            )
        )

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
        round_layout.addWidget(
            self._create_plot_explanation_label(
                [title for _, title in round_metrics]
            )
        )

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
        self._register_plot_canvas(self.agent_generation_area)
        self.agent_generation_area.setMinimumHeight(900)
        generation_layout.addWidget(self.agent_generation_area)
        generation_layout.addWidget(
            self._create_plot_explanation_label(
                [
                    "Profit Change vs Prev Gen",
                    "Info_Param per Agent",
                    "Qty_Aggression per Agent",
                    "Signal_Aggression per Agent",
                ]
            )
        )
        content_layout.addWidget(generation_group)

        round_group = QGroupBox("Agent Performance Across Rounds")
        round_layout = QVBoxLayout(round_group)
        self.agent_round_area = pg.GraphicsLayoutWidget()
        self._register_plot_canvas(self.agent_round_area)
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
        self._register_plot_theme(
            profit_change_plot,
            bottom_text="Generation",
            bottom_legend=[
                ("ZI", STRATEGY_COLORS["zi"]),
                ("Parameterised Informed", STRATEGY_COLORS["parameterised_informed"]),
            ],
            left_text="Value",
            title_text=profit_change_title,
        )
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
        ]
        self.agent_info_param_plot = self.agent_generation_area.addPlot(
            title=self._format_plot_title(generation_line_specs[0][1])
        )
        _style_plot(self.agent_info_param_plot)
        self._register_plot_theme(
            self.agent_info_param_plot,
            bottom_text="Generation",
            left_text=generation_line_specs[0][2],
            title_text=generation_line_specs[0][1],
        )
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
        round_layout.addWidget(
            self._create_plot_explanation_label(
                [title for _, title in round_metrics]
            )
        )

    def _initialise_metric_grid(self, graph_area, metric_specs, curve_store, x_label, plot_store=None):
        for idx, (metric_key, title) in enumerate(metric_specs):
            plot = graph_area.addPlot(title=self._format_plot_title(title))
            _style_plot(plot)
            self._register_plot_theme(
                plot,
                bottom_text=x_label,
                bottom_legend=[
                    ("ZI", STRATEGY_COLORS["zi"]),
                    ("Parameterised Informed", STRATEGY_COLORS["parameterised_informed"]),
                ],
                left_text="Value",
                title_text=title,
            )
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
            self._register_plot_theme(plot, bottom_text=x_label, left_text="Value", title_text=title)
            plot_store[metric_key] = (plot, title)
            if idx % 2 == 1:
                self.agent_round_area.nextRow()

    def _initialise_agent_generation_line_grid(self, plot_specs):
        for idx, (attr_name, title, y_label) in enumerate(plot_specs):
            plot = self.agent_generation_area.addPlot(title=self._format_plot_title(title))
            _style_plot(plot)
            self._register_plot_theme(plot, bottom_text="Generation", left_text=y_label, title_text=title)
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

    def load_comparison_data(self):
        if self.compare_worker is not None:
            QMessageBox.information(self, "Comparison In Progress", "A comparison load is already in progress.")
            return

        selected_ids = [
            item.data(Qt.UserRole)
            for item in self.comparison_experiment_list.selectedItems()
            if item.data(Qt.UserRole) is not None
        ]
        if len(selected_ids) < 2:
            QMessageBox.information(
                self,
                "Select More Runs",
                "Please select at least two experiments to compare.",
            )
            return

        db_path = self.input_db_path.text().strip()
        if not db_path:
            QMessageBox.warning(self, "Missing Database", "Please provide a DuckDB file path.")
            return

        self._comparison_selected_ids = [str(experiment_id) for experiment_id in selected_ids]
        self.btn_compare_runs.setEnabled(False)
        self.comparison_status_label.setText("Loading comparison data...")
        self.compare_worker = ComparisonLoaderWorker(db_path=db_path, experiment_ids=selected_ids)
        self.compare_worker.loaded.connect(self._apply_comparison_payload)
        self.compare_worker.error.connect(self._show_comparison_error)
        self.compare_worker.finished.connect(self._comparison_worker_finished)
        self.compare_worker.start()

    def clear_comparison(self):
        self._comparison_payload = None
        self._comparison_selected_ids = []
        self.comparison_experiment_list.clearSelection()
        self.comparison_summary_label.setText(
            "No comparison loaded. Select experiments from the left panel."
        )
        self.comparison_status_label.setText("Comparison cleared.")
        self._clear_comparison_plots()
        if self._last_payload is not None:
            self._populate_sql_tab(self._last_payload.get("sql_objects", {}))

    def _apply_sweep_defaults(self, sweep_name):
        defaults = SWEEP_DEFAULTS.get(sweep_name, SWEEP_DEFAULTS["population"])
        self.sweep_total_agents_input.setText(str(defaults["total_agents"]))
        self.sweep_generations_input.setText(str(defaults["n_generations"]))
        self.sweep_rounds_input.setText(str(defaults["n_rounds"]))
        self.sweep_fixed_drift_input.setText(str(defaults["fixed_drift"]))
        self.sweep_fixed_volatility_input.setText(str(defaults["fixed_volatility"]))
        self.sweep_population_values_input.setText(str(defaults["population_values"]))
        self.sweep_drift_values_input.setText(str(defaults["drift_values"]))
        self.sweep_volatility_values_input.setText(str(defaults["volatility_values"]))

    def _collect_sweep_settings(self):
        return {
            "db_path": self.input_db_path.text().strip(),
            "total_agents": int(self.sweep_total_agents_input.text().strip()),
            "n_generations": int(self.sweep_generations_input.text().strip()),
            "n_rounds": int(self.sweep_rounds_input.text().strip()),
            "fixed_drift": float(self.sweep_fixed_drift_input.text().strip()),
            "fixed_volatility": float(self.sweep_fixed_volatility_input.text().strip()),
            "population_values": self.sweep_population_values_input.text().strip(),
            "drift_values": self.sweep_drift_values_input.text().strip(),
            "volatility_values": self.sweep_volatility_values_input.text().strip(),
        }

    def run_sweep_comparison(self):
        if self.sweep_compare_worker is not None:
            QMessageBox.information(self, "Sweep In Progress", "A sweep comparison is already running.")
            return
        if self.run_worker is not None:
            QMessageBox.information(
                self,
                "Run In Progress",
                "Please wait for the current run to finish before starting a sweep comparison.",
            )
            return

        try:
            sweep_name = self.combo_sweep_type.currentText().strip()
            sweep_settings = self._collect_sweep_settings()
            if not sweep_settings["db_path"]:
                raise ValueError("Please provide a DuckDB file path before running a sweep.")
            _build_sweep_run_args(sweep_name, sweep_settings)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Sweep Settings", str(exc))
            return

        self._comparison_sweep_payload = {
            "sweep_name": sweep_name,
            "sweep_title": _build_sweep_title(sweep_name, sweep_settings),
            "settings": sweep_settings,
            "runs": [],
        }
        self._update_sweep_summary(self._comparison_sweep_payload)
        self._update_sweep_comparison_plots([])
        self.btn_run_sweep_comparison.setEnabled(False)
        self.btn_stop_run.setEnabled(True)
        self.sweep_status_label.setText("Launching sweep comparison...")
        self.sweep_compare_worker = SweepComparisonWorker(sweep_name, sweep_settings)
        self.sweep_compare_worker.progress.connect(self._handle_sweep_progress)
        self.sweep_compare_worker.completed.connect(self._apply_sweep_comparison_payload)
        self.sweep_compare_worker.error.connect(self._show_sweep_error)
        self.sweep_compare_worker.finished.connect(self._sweep_worker_finished)
        self.sweep_compare_worker.start()

    def clear_sweep_comparison(self):
        self._comparison_sweep_payload = None
        self._comparison_sweep_legend_items = []
        self.sweep_summary_label.setText(
            "No sweep comparison loaded. Configure a sweep from the left panel."
        )
        self.sweep_status_label.setText("Sweep comparison cleared.")
        self._clear_sweep_comparison_plots()

    def _handle_sweep_progress(self, payload):
        event = payload.get("event")
        if event == "sweep_started":
            self._comparison_sweep_payload = {
                "sweep_name": payload["sweep_name"],
                "sweep_title": payload["sweep_title"],
                "settings": dict(self._collect_sweep_settings()),
                "runs": [],
            }
            self._update_sweep_summary(self._comparison_sweep_payload)
            self.sweep_status_label.setText(
                f"Running {payload['sweep_name']} sweep across {payload['total_runs']} run(s)..."
            )
        elif event == "sweep_run_started":
            self.sweep_status_label.setText(
                f"Running sweep {payload['run_index']} of {payload['total_runs']}: {payload['run_label']}"
            )
        elif event == "generation_started":
            self.sweep_status_label.setText(
                f"Sweep run {payload['run_index']} of {payload['total_runs']} | "
                f"{payload['run_label']} generation {payload['generation_id']} of {payload['n_generations']}..."
            )
        elif event == "generation_completed":
            run_index = int(payload["run_index"]) - 1
            live_df = payload.get("live_generation_df", pd.DataFrame()).copy()
            if not live_df.empty and "generation_id" in live_df.columns and "generation" not in live_df.columns:
                live_df["generation"] = live_df["generation_id"].astype(int) - 1
            while len(self._comparison_sweep_payload["runs"]) <= run_index:
                self._comparison_sweep_payload["runs"].append({})
            self._comparison_sweep_payload["runs"][run_index] = {
                "label": payload["run_label"],
                "experiment_id": payload.get("experiment_id"),
                "data": live_df,
            }
            self._update_sweep_summary(self._comparison_sweep_payload)
            self._update_sweep_comparison_plots(self._comparison_sweep_payload["runs"])
            self.sweep_status_label.setText(
                f"Sweep run {payload['run_index']} of {payload['total_runs']} | "
                f"{payload['run_label']} completed generation {payload['generation_id']} of {payload['n_generations']}."
            )
        elif event == "sweep_run_completed":
            run_index = int(payload["run_index"]) - 1
            while len(self._comparison_sweep_payload["runs"]) <= run_index:
                self._comparison_sweep_payload["runs"].append({})
            self._comparison_sweep_payload["runs"][run_index] = {
                "label": payload["run_label"],
                "experiment_id": payload.get("experiment_id"),
                "data": payload.get("data", pd.DataFrame()).copy(),
            }
            self._update_sweep_summary(self._comparison_sweep_payload)
            self._update_sweep_comparison_plots(self._comparison_sweep_payload["runs"])
            self.sweep_status_label.setText(
                f"Sweep progress: {payload['completed_runs']}/{payload['total_runs']} completed "
                f"({payload['run_label']})."
            )

    def _apply_sweep_comparison_payload(self, payload):
        self._comparison_sweep_payload = payload
        self._update_sweep_summary(payload)
        self._update_sweep_comparison_plots(payload.get("runs", []))
        self._save_sweep_comparison_exports(payload.get("sweep_name", "sweep"))
        self.tabs.setCurrentWidget(self.comparison_tab)
        self.sweep_status_label.setText("Sweep comparison finished, saved to DuckDB, and exported to comparison_outputs.")
        self.refresh_data()

    def _show_sweep_error(self, message):
        self.sweep_status_label.setText("Sweep comparison failed.")
        QMessageBox.critical(self, "Sweep Comparison Error", message)

    def _sweep_worker_finished(self):
        self.btn_run_sweep_comparison.setEnabled(True)
        if self.run_worker is None:
            self.btn_stop_run.setEnabled(False)
        self.sweep_compare_worker = None

    def _comparison_worker_finished(self):
        self.btn_compare_runs.setEnabled(True)
        self.compare_worker = None

    def _show_comparison_error(self, message):
        self.comparison_status_label.setText("Comparison load failed.")
        QMessageBox.critical(self, "Comparison Error", message)

    def _apply_comparison_payload(self, payload):
        self._comparison_payload = payload
        experiments_df = payload.get("experiments", pd.DataFrame())
        comparison_df = payload.get("comparison_metrics", pd.DataFrame())
        comparison_runs = _comparison_payload_to_runs(experiments_df, comparison_df)
        self._update_comparison_summary(experiments_df, comparison_df)
        self._update_comparison_plots(comparison_df, experiments_df)
        self._update_sweep_summary(
            {
                "sweep_title": "Selected Run Comparison",
                "runs": comparison_runs,
                "settings": {},
            }
        )
        self._update_sweep_comparison_plots(comparison_runs)
        self._populate_sql_tab(payload.get("sql_objects", {}))
        self.tabs.setCurrentWidget(self.comparison_tab)
        self.comparison_status_label.setText("Comparison loaded.")

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

        self._reset_live_run_state()
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
                f"Running generation {payload['generation_id']} "
                f"of {payload['n_generations']}..."
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
                f"Completed generation {payload['generation_id']} "
                f"of {payload['n_generations']}."
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
            self.run_status_label.setText(
                f"Experiment {payload['experiment_id']} completed."
            )

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
            return
        if self.sweep_compare_worker is not None:
            self.sweep_compare_worker.terminate()
            self.sweep_status_label.setText("Sweep comparison stopped by user.")
            self.btn_stop_run.setEnabled(False)

    def _handle_run_error(self, message):
        self.run_status_label.setText("Run failed.")
        QMessageBox.critical(self, "Run Error", message)

    def _reset_live_run_state(self):
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
        self._update_microstructure_tab(payload)

        if payload.get("database_created"):
            self.status_label.setText("Created a new DuckDB database and loaded it.")
        else:
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
        self._populate_comparison_experiment_list(experiments_df)

    def _populate_comparison_experiment_list(self, experiments_df):
        selected_ids = set(self._comparison_selected_ids)
        self.comparison_experiment_list.clear()
        for _, row in experiments_df.iterrows():
            label = f"{row['experiment_name']} | {row['experiment_id']}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(row["experiment_id"]))
            self.comparison_experiment_list.addItem(item)
            if str(row["experiment_id"]) in selected_ids:
                item.setSelected(True)

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
        math_note = _resolve_plot_math_note(base_title)
        return (
            f"<span style='color: {self._theme['plot_foreground']};'>"
            f"{base_title} | {math_note} | smoothing {self.smoothing_window}"
            f"</span>"
        )

    def _update_all_plot_titles(self):
        self.plot_params.setTitle(self._format_plot_title(self.plot_params_title))
        self.plot_wealth.setTitle(self._format_plot_title(self.plot_wealth_title))
        self.plot_info_param.setTitle(self._format_plot_title(self.plot_info_param_title))
        self.plot_market.setTitle(self._format_plot_title(self.plot_market_title))
        self.plot_profit.setTitle(self._format_plot_title(self.plot_profit_title))
        self.plot_volume_share.setTitle(self._format_plot_title(self.plot_volume_share_title))
        self.micro_lob_plot.setTitle(self._format_plot_title(self.micro_lob_title))
        self.micro_candle_plot.setTitle(self._format_plot_title(self.micro_candle_title))
        self.micro_network_plot.setTitle(self._format_plot_title(self.micro_network_title))
        self.micro_pressure_plot.setTitle(self._format_plot_title(self.micro_pressure_title))
        self.micro_activity_plot.setTitle(self._format_plot_title(self.micro_activity_title))

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
        for plot, base_title in self.agent_round_plots.values():
            plot.setTitle(self._format_plot_title(base_title))
        for metric_key, plot in self.comparison_param_plots.items():
            plot.setTitle(self._format_plot_title(self.comparison_plot_titles[metric_key]))
        for metric_key, plot in self.comparison_wealth_plots.items():
            plot.setTitle(self._format_plot_title(self.comparison_plot_titles[metric_key]))
        for metric_key, plot in self.sweep_param_plots.items():
            plot.setTitle(self._format_plot_title(self.sweep_plot_titles[metric_key]))
        for metric_key, plot in self.sweep_diversity_plots.items():
            plot.setTitle(self._format_plot_title(self.sweep_plot_titles[metric_key]))
        for metric_key, plot in self.sweep_wealth_plots.items():
            plot.setTitle(self._format_plot_title(self.sweep_plot_titles[metric_key]))

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
            if self._comparison_payload is not None:
                self._update_all_plot_titles()
                self._update_comparison_plots(
                    self._comparison_payload.get("comparison_metrics", pd.DataFrame()),
                    self._comparison_payload.get("experiments", pd.DataFrame()),
                )
            if self._comparison_sweep_payload is not None:
                self._update_all_plot_titles()
                self._update_sweep_comparison_plots(
                    self._comparison_sweep_payload.get("runs", []),
                )
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
        self._update_microstructure_tab(self._last_payload)
        if self._comparison_payload is not None:
            self._update_comparison_plots(
                self._comparison_payload.get("comparison_metrics", pd.DataFrame()),
                self._comparison_payload.get("experiments", pd.DataFrame()),
            )
        if self._comparison_sweep_payload is not None:
            self._update_sweep_comparison_plots(
                self._comparison_sweep_payload.get("runs", []),
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
            return

        x = generations_df["generation_id"].tolist()
        self.line_qty.setData(x, self._smooth_series(generations_df["mean_qty_aggression"].tolist()))
        self.line_signal.setData(x, self._smooth_series(generations_df["mean_signal_aggression"].tolist()))

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
            color = STRATEGY_COLORS.get(strategy_type, (120, 120, 120))
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

        color = STRATEGY_COLORS["parameterised_informed"]
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

    def _update_microstructure_tab(self, payload):
        market_history_df = payload.get("market_history", pd.DataFrame())
        market_summary_df = payload.get("market_summary", pd.DataFrame())
        agent_round_df = payload.get("agent_round", pd.DataFrame())
        population_df = payload.get("population", pd.DataFrame())
        trade_execution_df = payload.get("trade_execution", pd.DataFrame())
        trade_link_df = trade_execution_df if not trade_execution_df.empty else _pair_trade_links_from_agent_round(agent_round_df)

        self._update_microstructure_round_selector(market_history_df, trade_link_df)
        self._update_microstructure_summary(market_history_df, trade_link_df, not trade_execution_df.empty)
        self._update_lob_snapshot_plot(agent_round_df, market_history_df)
        self._update_candlestick_plot(market_summary_df, market_history_df)
        self._update_trade_network_plot(trade_link_df, population_df, market_history_df)
        self._update_market_pressure_plot(market_history_df)
        self._update_market_activity_plot(market_history_df)

    def _update_microstructure_round_selector(self, market_history_df, trade_link_df):
        round_values = []
        if not market_history_df.empty and "round_number" in market_history_df.columns:
            round_values.extend(market_history_df["round_number"].dropna().astype(int).tolist())
        if not trade_link_df.empty and "round_number" in trade_link_df.columns:
            round_values.extend(trade_link_df["round_number"].dropna().astype(int).tolist())

        if not round_values:
            self._microstructure_round = None
            self.micro_round_slider.setEnabled(False)
            self.micro_round_slider.setMinimum(0)
            self.micro_round_slider.setMaximum(0)
            self.micro_round_slider.setValue(0)
            self.micro_round_slider_label.setText("Trade network round: unavailable")
            return

        min_round = int(min(round_values))
        max_round = int(max(round_values))
        selected_round = max_round if self._microstructure_round is None else int(self._microstructure_round)
        selected_round = max(min_round, min(max_round, selected_round))

        self.micro_round_slider.blockSignals(True)
        self.micro_round_slider.setEnabled(True)
        self.micro_round_slider.setMinimum(min_round)
        self.micro_round_slider.setMaximum(max_round)
        self.micro_round_slider.setValue(selected_round)
        self.micro_round_slider.blockSignals(False)
        self._microstructure_round = selected_round
        self.micro_round_slider_label.setText(
            f"Trade network round: {selected_round} of {max_round}"
        )

    def _on_micro_round_changed(self, value):
        self._microstructure_round = int(value)
        self.micro_round_slider_label.setText(
            f"Trade network round: {self._microstructure_round} of {self.micro_round_slider.maximum()}"
        )
        if self._last_payload is not None:
            market_history_df = self._last_payload.get("market_history", pd.DataFrame())
            agent_round_df = self._last_payload.get("agent_round", pd.DataFrame())
            population_df = self._last_payload.get("population", pd.DataFrame())
            trade_execution_df = self._last_payload.get("trade_execution", pd.DataFrame())
            trade_link_df = trade_execution_df if not trade_execution_df.empty else _pair_trade_links_from_agent_round(agent_round_df)
            self._update_trade_network_plot(trade_link_df, population_df, market_history_df)

    def _update_microstructure_summary(self, market_history_df, trade_link_df, using_persisted_links):
        if market_history_df.empty:
            self.microstructure_summary_label.setText(
                "No market microstructure data is available for the selected generation."
            )
            return

        spreads = (
            market_history_df["best_ask"].astype(float) - market_history_df["best_bid"].astype(float)
        )
        avg_spread = float(spreads.dropna().mean()) if spreads.notna().any() else float("nan")
        total_volume = float(market_history_df["volume"].sum()) if "volume" in market_history_df.columns else 0.0
        total_trades = int(market_history_df["n_trades"].sum()) if "n_trades" in market_history_df.columns else 0
        avg_active = (
            float(market_history_df["n_active_total"].mean())
            if "n_active_total" in market_history_df.columns and market_history_df["n_active_total"].notna().any()
            else float("nan")
        )
        network_edges = len(trade_link_df.groupby(["buyer_agent_id", "seller_agent_id"])) if not trade_link_df.empty else 0
        network_mode = "persisted trade links" if using_persisted_links else "reconstructed trade links"
        self.microstructure_summary_label.setText(
            "\n".join(
                [
                    f"Rounds loaded: {len(market_history_df)}",
                    f"Total matched volume: {total_volume:.3f}",
                    f"Total trade prints: {total_trades}",
                    f"Average quoted spread: {avg_spread:.4f}" if not np.isnan(avg_spread) else "Average quoted spread: unavailable",
                    f"Average active traders: {avg_active:.2f}" if not np.isnan(avg_active) else "Average active traders: unavailable",
                    f"Unique buyer-seller links: {network_edges}",
                    f"Trade network source: {network_mode}",
                ]
            )
        )

    def _update_lob_snapshot_plot(self, agent_round_df, market_history_df):
        self.micro_lob_plot.clear()
        self.micro_lob_plot.addLine(y=0, pen=pg.mkPen((150, 150, 150), width=1))
        if agent_round_df.empty:
            self.micro_lob_plot.setTitle(self._format_plot_title(self.micro_lob_title))
            return

        snapshot_round = int(agent_round_df["round_number"].max())
        round_df = agent_round_df[agent_round_df["round_number"] == snapshot_round].copy()
        round_df = round_df.dropna(subset=["limit_price", "order_qty"])
        if round_df.empty:
            self.micro_lob_plot.setTitle(
                self._format_plot_title(f"{self.micro_lob_title} | round {snapshot_round}")
            )
            return

        bids = (
            round_df[round_df["action"] == "buy"]
            .groupby("limit_price", as_index=False)["order_qty"]
            .sum()
            .sort_values("limit_price")
        )
        asks = (
            round_df[round_df["action"] == "sell"]
            .groupby("limit_price", as_index=False)["order_qty"]
            .sum()
            .sort_values("limit_price")
        )
        price_points = sorted(set(bids["limit_price"].tolist()) | set(asks["limit_price"].tolist()))
        if price_points:
            width = max(0.15, (min(np.diff(price_points)) * 0.8) if len(price_points) > 1 else 0.4)
        else:
            width = 0.4

        if not bids.empty:
            self.micro_lob_plot.addItem(
                pg.BarGraphItem(
                    x=bids["limit_price"].astype(float).tolist(),
                    height=bids["order_qty"].astype(float).tolist(),
                    width=width,
                    brush=(30, 144, 255, 180),
                    pen=pg.mkPen((30, 144, 255), width=1),
                )
            )
        if not asks.empty:
            self.micro_lob_plot.addItem(
                pg.BarGraphItem(
                    x=asks["limit_price"].astype(float).tolist(),
                    height=(-asks["order_qty"].astype(float)).tolist(),
                    width=width,
                    brush=(220, 20, 60, 180),
                    pen=pg.mkPen((220, 20, 60), width=1),
                )
            )

        if not market_history_df.empty:
            current_round_market = market_history_df[market_history_df["round_number"] == snapshot_round]
            if not current_round_market.empty:
                round_row = current_round_market.iloc[0]
                if pd.notna(round_row.get("p_t")):
                    self.micro_lob_plot.addLine(
                        x=float(round_row["p_t"]),
                        pen=pg.mkPen(self._theme["plot_outline"], width=2, style=Qt.DashLine),
                    )

        self.micro_lob_plot.setTitle(
            self._format_plot_title(f"{self.micro_lob_title} | round {snapshot_round}")
        )

    def _update_candlestick_plot(self, market_summary_df, market_history_df):
        self.micro_candle_plot.clear()
        if market_history_df.empty:
            self.micro_candle_plot.setTitle(self._format_plot_title(self.micro_candle_title))
            return

        merged_df = market_history_df.merge(
            market_summary_df,
            on="round_number",
            how="left",
            suffixes=("", "_summary"),
        ).sort_values("round_number")
        if merged_df.empty:
            self.micro_candle_plot.setTitle(self._format_plot_title(self.micro_candle_title))
            return

        previous_close = None
        for _, row in merged_df.iterrows():
            round_number = float(row["round_number"])
            close_price = (
                float(row["p_t"])
                if pd.notna(row.get("p_t"))
                else float(row.get("fundamental_price", 0.0))
            )
            open_price = close_price if previous_close is None else previous_close
            high_candidates = [
                value for value in [
                    row.get("max_sell"),
                    row.get("best_ask"),
                    open_price,
                    close_price,
                ] if pd.notna(value)
            ]
            low_candidates = [
                value for value in [
                    row.get("min_bid"),
                    row.get("best_bid"),
                    open_price,
                    close_price,
                ] if pd.notna(value)
            ]
            high_price = float(max(high_candidates)) if high_candidates else close_price
            low_price = float(min(low_candidates)) if low_candidates else close_price
            candle_color = (46, 139, 87) if close_price >= open_price else (220, 20, 60)
            self.micro_candle_plot.plot(
                x=[round_number, round_number],
                y=[low_price, high_price],
                pen=pg.mkPen(candle_color, width=1),
            )
            self.micro_candle_plot.plot(
                x=[round_number, round_number],
                y=[open_price, close_price],
                pen=pg.mkPen(candle_color, width=6),
            )
            previous_close = close_price

    def _update_trade_network_plot(self, trade_link_df, population_df, market_history_df):
        self.micro_network_plot.clear()
        if population_df.empty:
            self.micro_network_plot.setTitle(self._format_plot_title(self.micro_network_title))
            return

        population_df = population_df.sort_values("agent_id").reset_index(drop=True)
        agent_ids = population_df["agent_id"].astype(int).tolist()
        n_agents = len(agent_ids)
        angles = np.linspace(0, 2 * np.pi, num=n_agents, endpoint=False)
        radius = 10.0
        positions = {
            agent_id: (float(radius * np.cos(angle)), float(radius * np.sin(angle)))
            for agent_id, angle in zip(agent_ids, angles)
        }
        strategy_by_agent = {
            int(row["agent_id"]): str(row["strategy_type"])
            for _, row in population_df.iterrows()
        }

        snapshot_round = None
        if self._microstructure_round is not None:
            snapshot_round = int(self._microstructure_round)
        elif not market_history_df.empty and "round_number" in market_history_df.columns:
            snapshot_round = int(market_history_df["round_number"].max())
        elif not trade_link_df.empty and "round_number" in trade_link_df.columns:
            snapshot_round = int(trade_link_df["round_number"].max())

        if snapshot_round is not None and "round_number" in trade_link_df.columns:
            trade_link_df = trade_link_df[trade_link_df["round_number"] == snapshot_round].copy()

        buyer_agents = set()
        seller_agents = set()
        if not trade_link_df.empty:
            edge_df = (
                trade_link_df.groupby(["buyer_agent_id", "seller_agent_id"], as_index=False)["notional"]
                .sum()
                .sort_values("notional", ascending=False)
            )
            max_notional = float(edge_df["notional"].max()) if not edge_df.empty else 1.0
            for _, edge in edge_df.iterrows():
                buyer = int(edge["buyer_agent_id"])
                seller = int(edge["seller_agent_id"])
                buyer_agents.add(buyer)
                seller_agents.add(seller)
                if buyer not in positions or seller not in positions:
                    continue
                width = 1 + 5 * (float(edge["notional"]) / max_notional if max_notional > 0 else 0.0)
                x1, y1 = positions[buyer]
                x2, y2 = positions[seller]
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0
                dx = x2 - x1
                dy = y2 - y1
                self.micro_network_plot.plot(
                    x=[x1, mid_x],
                    y=[y1, mid_y],
                    pen=pg.mkPen((46, 139, 87, 180), width=width),
                )
                self.micro_network_plot.plot(
                    x=[mid_x, x2],
                    y=[mid_y, y2],
                    pen=pg.mkPen((220, 20, 60, 180), width=width),
                )
                arrow = pg.ArrowItem(
                    pos=(x2, y2),
                    angle=float(np.degrees(np.arctan2(dy, dx))),
                    headLen=14,
                    tipAngle=28,
                    baseAngle=20,
                    brush=(220, 20, 60),
                    pen=pg.mkPen((220, 20, 60), width=1),
                )
                self.micro_network_plot.addItem(arrow)

        scatter = pg.ScatterPlotItem(size=16)
        scatter_points = []
        for agent_id in agent_ids:
            strategy_type = strategy_by_agent.get(agent_id, "unknown")
            x_pos, y_pos = positions[agent_id]
            color = STRATEGY_COLORS.get(strategy_type, (180, 180, 180))
            if agent_id in buyer_agents:
                outline_color = (46, 139, 87)
                outline_width = 3
            elif agent_id in seller_agents:
                outline_color = (220, 20, 60)
                outline_width = 3
            else:
                outline_color = self._theme["plot_outline"]
                outline_width = 1
            scatter_points.append(
                {
                    "pos": (x_pos, y_pos),
                    "brush": pg.mkBrush(color),
                    "pen": pg.mkPen(outline_color, width=outline_width),
                }
            )
            label = pg.TextItem(
                text=str(agent_id),
                color=self._theme["plot_foreground"],
                anchor=(0.5, -0.4),
            )
            label.setPos(x_pos, y_pos)
            self.micro_network_plot.addItem(label)
        scatter.setData(spots=scatter_points)
        self.micro_network_plot.addItem(scatter)
        legend_y = radius + 4.0
        buy_label = pg.TextItem(text="Buyer outline", color=(46, 139, 87), anchor=(0, 0))
        buy_label.setPos(-radius, legend_y)
        self.micro_network_plot.addItem(buy_label)
        sell_label = pg.TextItem(text="Seller outline", color=(220, 20, 60), anchor=(0, 0))
        sell_label.setPos(-radius, legend_y + 1.6)
        self.micro_network_plot.addItem(sell_label)
        flow_label = pg.TextItem(
            text="Green line to red line + red arrow: buyer -> seller",
            color=self._theme["plot_foreground"],
            anchor=(0, 0),
        )
        flow_label.setPos(-radius, legend_y + 3.2)
        self.micro_network_plot.addItem(flow_label)
        self.micro_network_plot.enableAutoRange()
        if snapshot_round is not None:
            self.micro_network_plot.setTitle(
                self._format_plot_title(f"{self.micro_network_title} | round {snapshot_round}")
            )
        else:
            self.micro_network_plot.setTitle(self._format_plot_title(self.micro_network_title))

    def _update_market_pressure_plot(self, market_history_df):
        self.micro_pressure_plot.clear()
        if market_history_df.empty:
            self.micro_pressure_plot.setTitle(self._format_plot_title(self.micro_pressure_title))
            return

        x = market_history_df["round_number"].astype(float).tolist()
        if {"demand_at_p", "supply_at_p"}.issubset(market_history_df.columns):
            total_flow = market_history_df["demand_at_p"].astype(float) + market_history_df["supply_at_p"].astype(float)
            imbalance = np.where(
                total_flow.abs() > 1e-12,
                (market_history_df["demand_at_p"].astype(float) - market_history_df["supply_at_p"].astype(float)) / total_flow,
                0.0,
            )
            self.micro_pressure_plot.plot(
                x=x,
                y=self._smooth_series(imbalance.tolist()),
                pen=pg.mkPen((255, 140, 0), width=3),
            )
        if {"best_bid", "best_ask"}.issubset(market_history_df.columns):
            spread = (
                market_history_df["best_ask"].astype(float) - market_history_df["best_bid"].astype(float)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.micro_pressure_plot.plot(
                x=x,
                y=self._smooth_series(spread.tolist()),
                pen=pg.mkPen((30, 144, 255), width=3),
            )

    def _update_market_activity_plot(self, market_history_df):
        self.micro_activity_plot.clear()
        if market_history_df.empty:
            self.micro_activity_plot.setTitle(self._format_plot_title(self.micro_activity_title))
            return

        x = market_history_df["round_number"].astype(float).tolist()
        if "volume" in market_history_df.columns:
            self.micro_activity_plot.plot(
                x=x,
                y=self._smooth_series(market_history_df["volume"].astype(float).tolist()),
                pen=pg.mkPen((46, 139, 87), width=3),
            )
        if "n_trades" in market_history_df.columns:
            self.micro_activity_plot.plot(
                x=x,
                y=self._smooth_series(market_history_df["n_trades"].astype(float).tolist()),
                pen=pg.mkPen((220, 20, 60), width=3),
            )
        if "n_active_total" in market_history_df.columns:
            self.micro_activity_plot.plot(
                x=x,
                y=self._smooth_series(market_history_df["n_active_total"].astype(float).tolist()),
                pen=pg.mkPen((218, 165, 32), width=3),
            )

    def _clear_comparison_plots(self):
        for metric_key, plot in self.comparison_param_plots.items():
            plot.clear()
            plot.setTitle(self._format_plot_title(self.comparison_plot_titles[metric_key]))
            self._apply_theme_to_plot(plot)
        for metric_key, plot in self.comparison_wealth_plots.items():
            plot.clear()
            self._comparison_zero_lines[metric_key] = plot.addLine(
                y=0,
                pen=pg.mkPen(self._theme["plot_reference"], width=1, style=Qt.DashLine),
            )
            plot.setTitle(self._format_plot_title(self.comparison_plot_titles[metric_key]))
            self._apply_theme_to_plot(plot)
        self._comparison_legend_items = []
        empty_labels = _format_run_label_html([], text_color=self._theme["plot_foreground"])
        self.comparison_param_runs_label.setText(empty_labels)
        self.comparison_wealth_runs_label.setText(empty_labels)

    def _clear_sweep_comparison_plots(self):
        for metric_key, plot in self.sweep_param_plots.items():
            plot.clear()
            plot.setTitle(self._format_plot_title(self.sweep_plot_titles[metric_key]))
            self._apply_theme_to_plot(plot)
        for metric_key, plot in self.sweep_diversity_plots.items():
            plot.clear()
            plot.setTitle(self._format_plot_title(self.sweep_plot_titles[metric_key]))
            self._apply_theme_to_plot(plot)
        for metric_key, plot in self.sweep_wealth_plots.items():
            plot.clear()
            plot.addLine(
                y=0,
                pen=pg.mkPen(self._theme["plot_reference"], width=1, style=Qt.DashLine),
            )
            plot.setTitle(self._format_plot_title(self.sweep_plot_titles[metric_key]))
            self._apply_theme_to_plot(plot)
        self._comparison_sweep_legend_items = []
        empty_labels = _format_run_label_html([], text_color=self._theme["plot_foreground"])
        self.sweep_param_runs_label.setText(empty_labels)
        self.sweep_diversity_runs_label.setText(empty_labels)
        self.sweep_wealth_runs_label.setText(empty_labels)

    def _update_comparison_summary(self, experiments_df, comparison_df):
        if experiments_df.empty:
            self.comparison_summary_label.setText("No comparison data available.")
            return

        run_count = len(experiments_df)
        labels = [
            f"{row['experiment_name']} ({row['experiment_id']})"
            for _, row in experiments_df.iterrows()
        ]
        if comparison_df.empty:
            generation_summary = "No generation metrics found for the selected runs."
        else:
            generation_summary = (
                f"Generations loaded: {int(comparison_df['generation_id'].min())}"
                f" to {int(comparison_df['generation_id'].max())}"
            )
        self.comparison_summary_label.setText(
            "\n".join(
                [
                    f"Comparing {run_count} runs",
                    generation_summary,
                    "Runs:",
                    *labels,
                ]
            )
        )

    def _update_sweep_summary(self, payload):
        runs = payload.get("runs", [])
        settings = payload.get("settings", {})
        if not runs:
            self.sweep_summary_label.setText("No sweep comparison data available.")
            return

        generation_min = None
        generation_max = None
        run_labels = []
        for run in runs:
            run_labels.append(str(run.get("label", "Unknown run")))
            run_df = run.get("data", pd.DataFrame())
            if run_df.empty or "generation" not in run_df.columns:
                continue
            run_min = int(run_df["generation"].min())
            run_max = int(run_df["generation"].max())
            generation_min = run_min if generation_min is None else min(generation_min, run_min)
            generation_max = run_max if generation_max is None else max(generation_max, run_max)

        generation_summary = "Generations loaded: unavailable"
        if generation_min is not None and generation_max is not None:
            generation_summary = f"Generations loaded: {generation_min} to {generation_max}"

        self.sweep_summary_label.setText(
            "\n".join(
                [
                    f"Sweep: {payload.get('sweep_title', payload.get('sweep_name', 'unknown'))}",
                    f"Runs executed: {len(runs)}",
                    generation_summary,
                    f"Settings: gens={settings.get('n_generations')} | rounds={settings.get('n_rounds')} | total agents={settings.get('total_agents')}",
                    "Runs:",
                    *run_labels,
                ]
            )
        )

    def _update_comparison_plots(self, comparison_df, experiments_df):
        self._clear_comparison_plots()
        if comparison_df.empty or experiments_df.empty:
            return

        param_df = comparison_df.sort_values(["experiment_id", "generation_id"]).reset_index(drop=True)
        wealth_df = param_df

        labels_by_id = {
            str(row["experiment_id"]): f"{row['experiment_name']} | {str(row['experiment_id'])[:8]}"
            for _, row in experiments_df.iterrows()
        }
        legend_items = []

        run_groups = list(param_df.groupby("experiment_id", sort=False))
        gradient_colors = _sweep_colors(len(run_groups))
        for run_idx, (experiment_id, run_df) in enumerate(run_groups):
            color = gradient_colors[run_idx % len(gradient_colors)]
            transparent_color = QColor(*color)
            transparent_color.setAlpha(70)
            run_df = run_df.sort_values("generation_id")
            x = run_df["generation_id"].tolist()
            label = labels_by_id.get(str(experiment_id), str(experiment_id))
            legend_items.append((label, color))

            for metric_key, _ in COMPARISON_PARAM_SPECS:
                if metric_key not in run_df.columns:
                    continue
                values = run_df[metric_key].astype(float).tolist()
                self.comparison_param_plots[metric_key].plot(
                    x=x,
                    y=values,
                    pen=pg.mkPen(transparent_color, width=1),
                    name=None,
                    connect="finite",
                )
                self.comparison_param_plots[metric_key].plot(
                    x=x,
                    y=self._smooth_series(values),
                    pen=pg.mkPen(color, width=3),
                    name=label,
                    connect="finite",
                )

            wealth_run_df = wealth_df[wealth_df["experiment_id"] == experiment_id].sort_values("generation_id")
            if WEALTH_INFORMED_COL in wealth_run_df.columns and WEALTH_ZI_COL in wealth_run_df.columns:
                diff_values = (
                    wealth_run_df[WEALTH_INFORMED_COL].astype(float)
                    - wealth_run_df[WEALTH_ZI_COL].astype(float)
                ).tolist()
                wealth_x = wealth_run_df["generation_id"].tolist()
                wealth_plot = self.comparison_wealth_plots[COMPARISON_WEALTH_DIFF_SPEC[0]]
                wealth_plot.plot(
                    x=wealth_x,
                    y=diff_values,
                    pen=pg.mkPen(transparent_color, width=1),
                    name=None,
                    connect="finite",
                )
                wealth_plot.plot(
                    x=wealth_x,
                    y=self._smooth_series(diff_values),
                    pen=pg.mkPen(color, width=3),
                    name=label,
                    connect="finite",
                )

        self._comparison_legend_items = legend_items
        labels_html = _format_run_label_html(
            legend_items,
            text_color=self._theme["plot_foreground"],
        )
        self.comparison_param_runs_label.setText(labels_html)
        self.comparison_wealth_runs_label.setText(labels_html)

    def _update_sweep_comparison_plots(self, runs):
        self._clear_sweep_comparison_plots()
        if not runs:
            return

        param_runs = []
        for run in runs:
            if not run:
                param_runs.append(run)
                continue
            run_df = run.get("data", pd.DataFrame())
            if run_df.empty or "generation" not in run_df.columns:
                param_runs.append(run)
                continue
            param_runs.append(
                dict(run, data=run_df.sort_values("generation").reset_index(drop=True))
            )
        diversity_runs = param_runs
        wealth_runs = param_runs
        active_run_count = len([run for run in param_runs if run])
        colors = _sweep_colors(active_run_count)
        legend_items = []
        color_index = 0
        for run_index, run in enumerate(param_runs):
            if not run:
                continue
            label = str(run.get("label", f"Run {run_index + 1}"))
            run_df = run.get("data", pd.DataFrame())
            if run_df.empty or "generation" not in run_df.columns:
                continue

            run_df = run_df.sort_values("generation")
            x = run_df["generation"].astype(int).tolist()
            color = colors[color_index % len(colors)]
            color_index += 1
            transparent_color = QColor(*color)
            transparent_color.setAlpha(45)
            legend_items.append((label, color))

            for metric_key, _ in SWEEP_PARAM_SUBPLOTS:
                if metric_key not in run_df.columns:
                    continue
                values = run_df[metric_key].astype(float).tolist()
                self.sweep_param_plots[metric_key].plot(
                    x=x,
                    y=values,
                    pen=pg.mkPen(transparent_color, width=1),
                    connect="finite",
                )
                self.sweep_param_plots[metric_key].plot(
                    x=x,
                    y=self._smooth_series(values),
                    pen=pg.mkPen(color, width=3),
                    connect="finite",
                )

            diversity_run_df = diversity_runs[run_index].get("data", pd.DataFrame()) if run_index < len(diversity_runs) else pd.DataFrame()
            for metric_key, _ in SWEEP_STD_SUBPLOTS:
                if diversity_run_df.empty or metric_key not in diversity_run_df.columns:
                    continue
                values = diversity_run_df[metric_key].astype(float).tolist()
                diversity_x = diversity_run_df["generation"].astype(int).tolist()
                self.sweep_diversity_plots[metric_key].plot(
                    x=diversity_x,
                    y=values,
                    pen=pg.mkPen(transparent_color, width=1),
                    connect="finite",
                )
                self.sweep_diversity_plots[metric_key].plot(
                    x=diversity_x,
                    y=self._smooth_series(values),
                    pen=pg.mkPen(color, width=3),
                    connect="finite",
                )

            wealth_run_df = wealth_runs[run_index].get("data", pd.DataFrame()) if run_index < len(wealth_runs) else pd.DataFrame()
            if not wealth_run_df.empty and WEALTH_INFORMED_COL in wealth_run_df.columns and WEALTH_ZI_COL in wealth_run_df.columns:
                diff_values = (
                    wealth_run_df[WEALTH_INFORMED_COL].astype(float)
                    - wealth_run_df[WEALTH_ZI_COL].astype(float)
                ).tolist()
                wealth_x = wealth_run_df["generation"].astype(int).tolist()
                wealth_plot = self.sweep_wealth_plots[COMPARISON_WEALTH_DIFF_SPEC[0]]
                wealth_plot.plot(
                    x=wealth_x,
                    y=self._smooth_series(diff_values),
                    pen=pg.mkPen(color, width=3),
                    connect="finite",
                )

        self._comparison_sweep_legend_items = legend_items
        labels_html = _format_run_label_html(
            legend_items,
            text_color=self._theme["plot_foreground"],
        )
        self.sweep_param_runs_label.setText(labels_html)
        self.sweep_diversity_runs_label.setText(labels_html)
        self.sweep_wealth_runs_label.setText(labels_html)

    def _save_sweep_comparison_exports(self, sweep_name):
        SWEEP_COMPARISON_OUTPUT_DIR.mkdir(exist_ok=True)
        exports = [
            (self.sweep_param_area, SWEEP_COMPARISON_OUTPUT_DIR / f"{sweep_name}_params.png"),
            (self.sweep_diversity_area, SWEEP_COMPARISON_OUTPUT_DIR / f"{sweep_name}_diversity.png"),
            (self.sweep_wealth_area, SWEEP_COMPARISON_OUTPUT_DIR / f"{sweep_name}_wealth.png"),
        ]
        for widget, path in exports:
            widget.grab().save(str(path), "PNG")


if __name__ == "__main__":
    print("[gui] starting application")
    try:
        app = QApplication(sys.argv)
        print("[gui] QApplication created")
        window = CommandCenter()
        print("[gui] CommandCenter created")
        window.show()
        print("[gui] window shown")
        exit_code = app.exec()
        print(f"[gui] event loop exited with code {exit_code}")
        sys.exit(exit_code)
    except Exception as exc:
        print(f"[gui] startup failed: {exc}")
        raise
