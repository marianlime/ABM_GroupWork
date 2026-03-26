from pathlib import Path
from uuid import uuid4

from PySide6.QtWidgets import QApplication

from Database.database_creation import create_database
from gui import CommandCenter, DatabaseLoaderWorker
from Database.SQL_Functions import (
    insert_agent_population,
    insert_agent_round_rows,
    insert_experiment_row,
    insert_fundamental_series,
    insert_gbm_config_row,
    insert_generation_row,
    insert_market_round_rows,
    insert_trade_execution_rows,
    update_generation_param_means,
)


class _StubAgent:
    def __init__(self, trader_type, info_param, cash, shares, strategy_params=None):
        self.trader_type = trader_type
        self.info_param = info_param
        self.cash = cash
        self.shares = shares
        self.strategy_params = strategy_params or {}


def _build_sample_db(db_path: Path):
    create_database(str(db_path))

    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        experiment_id = "01TESTEXPERIMENTID000000000"

        insert_experiment_row(
            con,
            experiment_id,
            "GUI Test Experiment",
            "integration-test",
            "2026-03-22 12:00:00",
            None,
            "gui smoke test",
            2,
            2,
            "GBM",
            "3.12",
            "test-version",
            2,
            1000.0,
            10,
            "call_auction",
            "maximum_volume_minimum_imbalance",
            "proportional_rationing",
            "previous_price_proximity",
            0.0,
            "evenly_spaced",
            {"low": 0.0, "high": 1.0},
            "lognormal",
            "truncation",
            {"mutation_rate": 0.05},
            0.0,
        )

        for generation_id, qty_mean in [(1, 0.55), (2, 0.65)]:
            insert_generation_row(
                con,
                experiment_id,
                generation_id,
                f"2026-03-22 12:00:0{generation_id}",
                f"2026-03-22 12:00:1{generation_id}",
                "COMPLETED",
                100.0,
            )
            update_generation_param_means(
                con,
                experiment_id,
                generation_id,
                qty_mean,
                0.45 + 0.05 * generation_id,
                0.10 + 0.02 * generation_id,
                0.80 + 0.03 * generation_id,
            )
            insert_gbm_config_row(
                con,
                experiment_id,
                generation_id,
                100.0,
                0.2,
                0.05,
                f"seed-{generation_id}",
            )
            insert_fundamental_series(
                con,
                experiment_id,
                generation_id,
                ((0, 100.0 + generation_id), (1, 101.0 + generation_id)),
            )
            insert_market_round_rows(
                con,
                [
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 0,
                        "p_t": 100.5 + generation_id,
                        "best_bid": 100.0 + generation_id,
                        "best_ask": 101.0 + generation_id,
                        "volume": 5.0,
                        "n_trades": 1,
                        "demand_at_p": 5.0,
                        "supply_at_p": 5.0,
                        "n_active_buyers": 1,
                        "n_active_sellers": 1,
                        "n_active_total": 2,
                        "bid_depth_total": 5.0,
                        "ask_depth_total": 5.0,
                        "price_levels_bid": 1,
                        "price_levels_ask": 1,
                    },
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 1,
                        "p_t": 101.5 + generation_id,
                        "best_bid": 101.0 + generation_id,
                        "best_ask": 102.0 + generation_id,
                        "volume": 6.0,
                        "n_trades": 1,
                        "demand_at_p": 6.0,
                        "supply_at_p": 6.0,
                        "n_active_buyers": 1,
                        "n_active_sellers": 1,
                        "n_active_total": 2,
                        "bid_depth_total": 6.0,
                        "ask_depth_total": 6.0,
                        "price_levels_bid": 1,
                        "price_levels_ask": 1,
                    },
                ],
            )

            agents = {
                1: _StubAgent(
                    "parameterised_informed",
                    0.3,
                    500.0,
                    5.0,
                    {
                        "qty_aggression": qty_mean,
                        "signal_aggression": 0.45 + 0.05 * generation_id,
                    },
                ),
                2: _StubAgent("zi", 0.0, 500.0, 5.0),
            }
            insert_agent_population(con, experiment_id, generation_id, agents)
            insert_agent_round_rows(
                con,
                [
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 0,
                        "agent_id": 1,
                        "signal": 101.0,
                        "signal_error": 0.2,
                        "action": "buy",
                        "limit_price": 101.0,
                        "order_qty": 2.0,
                        "aggressiveness": 0.7,
                        "executed_qty": 1.0,
                        "executed_price_avg": 101.0,
                        "fill_ratio": 0.5,
                        "is_filled": False,
                        "is_partial": True,
                        "cash_start": 500.0,
                        "inventory_start": 5.0,
                        "cash_end": 400.0,
                        "inventory_end": 6.0,
                    },
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 0,
                        "agent_id": 2,
                        "signal": None,
                        "signal_error": None,
                        "action": "sell",
                        "limit_price": 101.0,
                        "order_qty": 1.0,
                        "aggressiveness": 0.2,
                        "executed_qty": 1.0,
                        "executed_price_avg": 101.0,
                        "fill_ratio": 1.0,
                        "is_filled": True,
                        "is_partial": False,
                        "cash_start": 500.0,
                        "inventory_start": 5.0,
                        "cash_end": 600.0,
                        "inventory_end": 4.0,
                    },
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 1,
                        "agent_id": 1,
                        "signal": 102.0,
                        "signal_error": 0.1,
                        "action": "buy",
                        "limit_price": 102.0,
                        "order_qty": 2.0,
                        "aggressiveness": 0.8,
                        "executed_qty": 2.0,
                        "executed_price_avg": 102.0,
                        "fill_ratio": 1.0,
                        "is_filled": True,
                        "is_partial": False,
                        "cash_start": 400.0,
                        "inventory_start": 6.0,
                        "cash_end": 300.0,
                        "inventory_end": 7.0,
                    },
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 1,
                        "agent_id": 2,
                        "signal": None,
                        "signal_error": None,
                        "action": "sell",
                        "limit_price": 102.0,
                        "order_qty": 1.0,
                        "aggressiveness": 0.3,
                        "executed_qty": 1.0,
                        "executed_price_avg": 102.0,
                        "fill_ratio": 1.0,
                        "is_filled": True,
                        "is_partial": False,
                        "cash_start": 600.0,
                        "inventory_start": 4.0,
                        "cash_end": 700.0,
                        "inventory_end": 3.0,
                    },
                ],
            )
            insert_trade_execution_rows(
                con,
                [
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 0,
                        "trade_id": 1,
                        "buyer_agent_id": 1,
                        "seller_agent_id": 2,
                        "price": 101.0,
                        "quantity": 1.0,
                        "notional": 101.0,
                    },
                    {
                        "experiment_id": experiment_id,
                        "generation_id": generation_id,
                        "round_number": 1,
                        "trade_id": 1,
                        "buyer_agent_id": 1,
                        "seller_agent_id": 2,
                        "price": 102.0,
                        "quantity": 1.0,
                        "notional": 102.0,
                    },
                ],
            )
    finally:
        con.close()

    return experiment_id


def test_database_loader_worker_reads_full_payload():
    db_path = Path(f"test_gui_{uuid4().hex}.duckdb")
    try:
        experiment_id = _build_sample_db(db_path)

        worker = DatabaseLoaderWorker(str(db_path))
        payload = worker._load_payload()

        assert payload["selected_experiment_id"] == experiment_id
        assert payload["selected_generation_id"] == 2
        assert payload["experiments"].shape[0] == 1
        assert payload["generations"].shape[0] == 2
        assert payload["wealth_history"].shape[0] == 4
        assert payload["mean_info_param"].shape[0] == 4
        assert payload["strategy_generation"].shape[0] == 4
        assert payload["agent_strategy_param_history"].shape[0] == 2
        assert "experiments" in payload["sql_objects"]
        assert "strategy_performance_per_generation" in payload["sql_objects"]
        assert payload["sql_objects"]["experiments"]["display_name"] == "Experiments"
        assert payload["market_history"].shape[0] == 2
        assert payload["population"].shape[0] == 2
        assert payload["agent_round"].shape[0] == 4
        assert payload["trade_execution"].shape[0] == 2
    finally:
        if db_path.exists():
            db_path.unlink()


def test_command_center_populates_widgets_from_payload(monkeypatch):
    db_path = Path(f"test_gui_{uuid4().hex}.duckdb")
    try:
        experiment_id = _build_sample_db(db_path)
        payload = DatabaseLoaderWorker(str(db_path))._load_payload()

        monkeypatch.setattr("gui.DEFAULT_DB_PATH", str(db_path))

        app = QApplication.instance() or QApplication([])

        original_refresh = CommandCenter.refresh_data
        CommandCenter.refresh_data = lambda self: None
        try:
            window = CommandCenter()
        finally:
            CommandCenter.refresh_data = original_refresh

        try:
            window._apply_payload(payload)

            assert window.combo_experiment.count() == 1
            assert window.combo_generation.count() == 2
            assert experiment_id in window.summary_label.text()
            assert window.table_population.rowCount() == 2
            assert window.table_market.rowCount() == 2
            assert window.table_agent_round.rowCount() == 4
            assert window.sql_tab.count() >= 6
            assert "Table" in window.sql_tab.tabText(0) or "View" in window.sql_tab.tabText(0)
            assert len(window.line_qty.getData()[0]) == 2
            assert len(window.line_informed_wealth.getData()[0]) == 2
            assert len(window.line_info_param_informed.getData()[0]) == 2
            assert len(window.line_mid_price.getData()[0]) == 2
            assert len(window.line_profit_informed.getData()[0]) == 2
            assert len(window.line_volume_informed.getData()[0]) == 2
            assert len(window.agent_info_param_plot.listDataItems()) == 1
            assert len(window.agent_qty_aggression_plot.listDataItems()) == 1
            assert len(window.agent_signal_aggression_plot.listDataItems()) == 1
            assert window.microstructure_summary_label.text()
        finally:
            window.close()
            app.processEvents()
    finally:
        if db_path.exists():
            db_path.unlink()


def test_command_center_start_new_run_builds_config(monkeypatch):
    class _Signal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

    captured = {}

    class _StubRunWorker:
        def __init__(self, config_overrides):
            captured["config_overrides"] = config_overrides
            self.progress = _Signal()
            self.completed = _Signal()
            self.error = _Signal()
            self.finished = _Signal()

        def start(self):
            captured["started"] = True

    app = QApplication.instance() or QApplication([])

    original_refresh = CommandCenter.refresh_data
    CommandCenter.refresh_data = lambda self: None
    monkeypatch.setattr("gui.ExperimentRunnerWorker", _StubRunWorker)
    try:
        window = CommandCenter()
    finally:
        CommandCenter.refresh_data = original_refresh

    try:
        window.input_db_path.setText("my_runs.duckdb")
        window.run_input_name.setText("GUI Run")
        window.run_input_type.setText("manual")
        window.run_input_notes.setPlainText("started from gui")
        window.run_input_seed.setText("12345")
        window.run_input_generations.setText("3")
        window.run_input_rounds.setText("4")
        window.run_input_zi.setText("7")
        window.run_input_informed.setText("8")
        window.run_input_cash.setText("1500")
        window.run_input_shares.setText("12")
        window.run_input_s0.setText("101")
        window.run_input_volatility.setText("0.3")
        window.run_input_drift.setText("0.07")
        window.run_input_mutation.setText("0.11")

        window.start_new_run()

        assert captured["started"] is True
        assert captured["config_overrides"]["db_path"] == "my_runs.duckdb"
        assert captured["config_overrides"]["experiment_name"] == "GUI Run"
        assert captured["config_overrides"]["n_generations"] == 3
        assert captured["config_overrides"]["n_rounds"] == 4
        assert captured["config_overrides"]["n_zi_agents"] == 7
        assert captured["config_overrides"]["n_parameterised_agents"] == 8
        assert captured["config_overrides"]["algorithm_params"]["mutation_rate"] == 0.11
        assert window.tabs.count() == 6
    finally:
        window.close()
        app.processEvents()
