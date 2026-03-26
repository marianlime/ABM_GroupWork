import numpy as np
import pytest

from Simulation.gbm import rng_from_string, simulate_gbm


def test_rng_same_seed_same_output():
    r1 = rng_from_string("seed_42")
    r2 = rng_from_string("seed_42")
    assert r1.random() == r2.random()


def test_rng_different_seeds():
    r1 = rng_from_string("alpha")
    r2 = rng_from_string("beta")
    assert r1.random() != r2.random()


def test_rng_accepts_int():
    r = rng_from_string(12345)
    assert hasattr(r, "random")


def test_gbm_length():
    path = simulate_gbm(S_0=100, volatility=0.2, drift=0.05, n_rounds=10, seed="abc")
    assert len(path) == 11


def test_gbm_starts_at_S0():
    path = simulate_gbm(S_0=50, volatility=0.1, drift=0.0, n_rounds=5, seed="x")
    assert np.isclose(path[0], 50.0)


def test_gbm_always_positive():
    path = simulate_gbm(S_0=100, volatility=0.5, drift=-0.1, n_rounds=200, seed="pos")
    assert np.all(path > 0)


def test_gbm_reproducible():
    p1 = simulate_gbm(S_0=100, volatility=0.2, drift=0.05, n_rounds=20, seed="rep")
    p2 = simulate_gbm(S_0=100, volatility=0.2, drift=0.05, n_rounds=20, seed="rep")
    assert np.allclose(p1, p2)


def test_gbm_different_seeds_differ():
    p1 = simulate_gbm(S_0=100, volatility=0.2, drift=0.05, n_rounds=20, seed="aaa")
    p2 = simulate_gbm(S_0=100, volatility=0.2, drift=0.05, n_rounds=20, seed="bbb")
    assert not np.allclose(p1, p2)


def test_gbm_zero_vol_is_deterministic():
    mu = 0.05
    path = simulate_gbm(S_0=100, volatility=0.0, drift=mu, n_rounds=5, seed="det")
    for t in range(6):
        assert np.isclose(path[t], 100 * np.exp(mu * t), rtol=1e-10)


def test_gbm_negative_S0():
    with pytest.raises(ValueError):
        simulate_gbm(S_0=-1, volatility=0.2, drift=0.0, n_rounds=10)


def test_gbm_zero_S0():
    with pytest.raises(ValueError):
        simulate_gbm(S_0=0, volatility=0.2, drift=0.0, n_rounds=10)


def test_gbm_negative_vol():
    with pytest.raises(ValueError):
        simulate_gbm(S_0=100, volatility=-0.1, drift=0.0, n_rounds=10)


def test_gbm_inf_drift():
    with pytest.raises(ValueError):
        simulate_gbm(S_0=100, volatility=0.2, drift=float("inf"), n_rounds=10)


def test_gbm_zero_rounds():
    with pytest.raises(ValueError):
        simulate_gbm(S_0=100, volatility=0.2, drift=0.0, n_rounds=0)
