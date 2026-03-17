import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from sim.runner import play_game


COMMON = dict(
    n_rounds=5,
    total_initial_shares=60,
    total_initial_cash=6000,
    run_id="TEST_RUNNER_01",
    noise_parameter_distribution_type="uniform",
    distribution_data={"low": 0.05, "high": 0.3},
    signal_generator_noise_distribution="lognormal",
    S0=100,
    fundamental_path=np.linspace(100, 105, 6),
    seed="runner_seed",
)

POP = [{"trader_type": "zi"}] * 4 + [{"trader_type": "parameterised_informed"}] * 2


@patch("sim.runner.update_run_progress")
@patch("sim.runner.insert_agent_population")
def test_returns_scores_and_game(mock_ins, mock_upd):
    scores, game = play_game(con=MagicMock(), population_spec=POP, **COMMON)
    assert len(scores) == 6
    for aid, w in scores:
        assert isinstance(w, float)


@patch("sim.runner.update_run_progress")
@patch("sim.runner.insert_agent_population")
def test_reaches_final_round(mock_ins, mock_upd):
    _, game = play_game(con=MagicMock(), population_spec=POP, **COMMON)
    assert game.current_round == 5


@patch("sim.runner.update_run_progress")
@patch("sim.runner.insert_agent_population")
def test_db_insert_called(mock_ins, mock_upd):
    play_game(con=MagicMock(), population_spec=POP, **COMMON)
    mock_ins.assert_called_once()


@patch("sim.runner.update_run_progress")
@patch("sim.runner.insert_agent_population")
def test_marked_completed(mock_ins, mock_upd):
    play_game(con=MagicMock(), population_spec=POP, **COMMON)
    assert "COMPLETED" in str(mock_upd.call_args)
