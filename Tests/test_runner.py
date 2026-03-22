import numpy as np

from sim.runner import play_game


COMMON = dict(
    n_rounds=5,
    total_initial_shares=60,
    total_initial_cash=6000,
    experiment_id="TEST_EXP_01",
    generation_id=1,
    info_param_distribution_type="uniform",
    distribution_data={"low": 0.05, "high": 0.3},
    signal_generator_noise_distribution="lognormal",
    S0=100,
    fundamental_path=np.linspace(100, 105, 6),
    seed="runner_seed",
)

POP = [{"trader_type": "zi"}] * 4 + [{"trader_type": "parameterised_informed"}] * 2


def test_returns_scores_and_game():
    scores, game = play_game(population_spec=POP, **COMMON)
    assert len(scores) == 6
    for _, wealth in scores:
        assert isinstance(wealth, float)


def test_reaches_final_round():
    _, game = play_game(population_spec=POP, **COMMON)
    assert game.current_round == 5


def test_returns_nonempty_agent_round_records():
    _, game = play_game(population_spec=POP, **COMMON)
    assert len(game.agent_round_records) > 0
