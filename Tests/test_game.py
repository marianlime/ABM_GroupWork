# Run via pytest in terminal

import numpy as np
import pytest

from Simulation.game import Game, _stable_seed


def _make_game(pop=None, n_rounds=5):
    """Build a Game with sensible defaults for testing."""
    if pop is None:
        pop = [{"trader_type": "zi"}] * 4 + [{"trader_type": "parameterised_informed"}] * 2

    path = np.linspace(100, 100 + n_rounds, n_rounds + 1)

    return Game(
        population_spec=pop,
        n_rounds=n_rounds,
        total_initial_shares=60,
        total_initial_cash=6000,
        experiment_id="TEST_EXP_01",
        generation_id=1,
        info_param_distribution_type="uniform",
        distribution_data={"low": 0.05, "high": 0.3},
        signal_generator_noise_distribution="lognormal",
        S0=100,
        fundamental_path=path,
        seed="test",
    )


# Seed helpers should be deterministic and independent

class TestSeedHelpers:

    def test_noise_seed_stable(self):
        assert _stable_seed("noise", "EXP", 1) == _stable_seed("noise", "EXP", 1)

    def test_zi_seed_stable(self):
        assert _stable_seed("zi", "EXP", 1) == _stable_seed("zi", "EXP", 1)

    def test_noise_and_zi_seeds_differ(self):
        assert _stable_seed("noise", "EXP", 1) != _stable_seed("zi", "EXP", 1)


# Game construction

class TestGameInit:

    def test_agent_count(self):
        g = _make_game()
        assert len(g.agents) == 6

    def test_cash_split(self):
        g = _make_game()
        for a in g.agents.values():
            assert np.isclose(a.cash, 1000)  # 6000 / 6

    def test_shares_split(self):
        g = _make_game()
        for a in g.agents.values():
            assert np.isclose(a.shares, 10)  # 60 / 6

    def test_round_starts_at_zero(self):
        assert _make_game().current_round == 0

    def test_empty_pop_raises(self):
        with pytest.raises(ValueError):
            _make_game(pop=[])

    def test_bad_trader_type_raises(self):
        with pytest.raises(ValueError):
            _make_game(pop=[{"trader_type": "wizard"}])

    def test_evolved_info_param_used(self):
        pop = [{"trader_type": "parameterised_informed", "info_param": 0.77}]
        g = _make_game(pop=pop)
        assert np.isclose(g.agents[1].info_param, 0.77)

    def test_zi_vs_informed_caches(self):
        pop = [{"trader_type": "zi"}] * 3 + [{"trader_type": "parameterised_informed"}] * 2
        g = _make_game(pop=pop)
        assert len(g._zi_agent_ids) == 3 and len(g._informed_agent_ids) == 2


# Batch ZI order generation

class TestBatchZIOrders:

    def test_returns_one_entry_per_zi_agent(self):
        g = _make_game(pop=[{"trader_type": "zi"}] * 5)
        orders = g._batch_zi_orders(prev_price=100)
        assert len(orders) == 5

    def test_empty_when_no_zi_agents(self):
        g = _make_game(pop=[{"trader_type": "parameterised_informed"}] * 3)
        assert g._batch_zi_orders(prev_price=100) == {}

    def test_prices_always_positive(self):
        g = _make_game(pop=[{"trader_type": "zi"}] * 20)
        orders = g._batch_zi_orders(prev_price=100)
        for o in orders.values():
            if o is not None:
                assert o["Price"] >= 0.01


# Order gathering and market clearing

class TestGatherOrdersAndClear:

    def test_returns_triple(self):
        g = _make_game()
        price, vol, trades = g.gather_orders_and_clear(0)
        assert isinstance(trades, list)

    def test_records_appended(self):
        g = _make_game()
        g.gather_orders_and_clear(0)
        assert len(g.market_round_records) == 1
        assert len(g.agent_round_records) == len(g.agents)

    def test_no_path_raises(self):
        g = _make_game()
        g.fundamental_path = None
        with pytest.raises(ValueError):
            g.gather_orders_and_clear(0)

    def test_round_too_large_raises(self):
        g = _make_game(n_rounds=5)
        with pytest.raises(IndexError):
            g.gather_orders_and_clear(100)


# Portfolio updates after trades

class TestUpdatePortfolio:

    def test_buy_adds_shares(self):
        g = _make_game()
        a = g.agents[1]
        before = a.shares
        g.update_portfolio({"agent_id": 1, "quantity": 2, "price": 100, "action": "buy"})
        assert np.isclose(a.shares, before + 2)

    def test_buy_removes_cash(self):
        g = _make_game()
        a = g.agents[1]
        before = a.cash
        g.update_portfolio({"agent_id": 1, "quantity": 2, "price": 100, "action": "buy"})
        assert a.cash < before

    def test_sell_removes_shares(self):
        g = _make_game()
        a = g.agents[1]
        before = a.shares
        g.update_portfolio({"agent_id": 1, "quantity": 3, "price": 50, "action": "sell"})
        assert np.isclose(a.shares, before - 3)

    def test_bad_action_raises(self):
        g = _make_game()
        with pytest.raises(ValueError):
            g.update_portfolio({"agent_id": 1, "quantity": 1, "price": 100, "action": "hold"})


# Terminal liquidation

class TestLiquidateAssets:

    def test_value_equals_shares_times_terminal_plus_cash(self):
        g = _make_game()
        a = g.agents[1]
        terminal = float(g.fundamental_path[-1])
        assert np.isclose(g.liquidate_assets(1), terminal * a.shares + a.cash)
