import numpy as np
import pytest

from sim.noise_signal import assign_noise_parameter_set


def test_assign_noise_uniform_shape_and_range():
    arr = assign_noise_parameter_set(100, "uniform", {"low": 0.0, "high": 0.5})
    assert len(arr) == 100
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 0.5)


def test_assign_noise_evenly_spaced_values():
    arr = assign_noise_parameter_set(5, "evenly_spaced", {"low": 0.0, "high": 1.0})
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(arr, expected)


def test_assign_noise_bimodal_shape_and_clipping():
    arr = assign_noise_parameter_set(100, "bimodal", {
        "group_a_mean": 0.2,
        "group_a_std":  0.05,
        "group_b_mean": 0.8,
        "group_b_std":  0.05,
    })
    assert len(arr) == 100
    assert np.all(arr >= 0.01)
    assert np.all(arr <= 1.5)


def test_assign_noise_skewed_positive():
    arr = assign_noise_parameter_set(100, "skewed", {"mean": 0.2, "sigma": 0.5})
    assert len(arr) == 100
    assert np.all(arr > 0)


def test_assign_noise_uniform_reproducible_with_seed():
    arr1 = assign_noise_parameter_set(10, "uniform", {"low": 0.0, "high": 0.5}, seed=42)
    arr2 = assign_noise_parameter_set(10, "uniform", {"low": 0.0, "high": 0.5}, seed=42)
    assert np.allclose(arr1, arr2)


def test_assign_noise_bimodal_odd_num_agents():
    arr = assign_noise_parameter_set(7, "bimodal", {
        "group_a_mean": 0.2, "group_a_std": 0.05,
        "group_b_mean": 0.8, "group_b_std": 0.05,
    })
    assert len(arr) == 7


def test_assign_noise_invalid_type_raises():
    with pytest.raises(ValueError):
        assign_noise_parameter_set(10, "bad_type", {})
