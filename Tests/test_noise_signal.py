import numpy as np
import pytest

from sim.noise_signal import assign_noise_parameter_set, signal_generator

def test_assign_noise_uniform_shape_and_range():
    # Generate 100 uniform noise parameters
    arr = assign_noise_parameter_set(100, "uniform")

    # Check the output has the correct number of elements
    assert len(arr) == 100

    # Uniform branch should only generate values in [0.0, 0.5]
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 0.5)


def test_assign_noise_evenly_spaced_values():
    # Evenly spaced branch should match np.linspace exactly
    arr = assign_noise_parameter_set(5, "evenly_spaced")
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Compare generated values to the expected evenly spaced sequence
    assert np.allclose(arr, expected)


def test_assign_noise_bimodal_shape_and_clipping():
    # Generate bimodal noise parameters for an even number of agents
    arr = assign_noise_parameter_set(100, "bimodal")

    # Check correct output size
    assert len(arr) == 100

    # Values should be clipped to the valid interval [0.01, 1.5]
    assert np.all(arr >= 0.01)
    assert np.all(arr <= 1.5)


def test_assign_noise_skewed_positive():
    # Lognormal/skewed values should always be positive
    arr = assign_noise_parameter_set(100, "skewed")

    # Check correct output size
    assert len(arr) == 100

    # All values should be strictly greater than zero
    assert np.all(arr > 0)


def test_assign_noise_uniform_reproducible_with_seed():
    # Fix random seed so the random draw is reproducible
    np.random.seed(42)
    arr1 = assign_noise_parameter_set(10, "uniform")

    # Reset the seed and generate again
    np.random.seed(42)
    arr2 = assign_noise_parameter_set(10, "uniform")

    # Both arrays should be identical because the seed is the same
    assert np.allclose(arr1, arr2)


def test_signal_generator_lognormal_positive():
    # Lognormal multiplicative noise should produce a positive signal
    out = signal_generator(0.2, 100.0, bias=0.0, noise_distribution="lognormal")

    # Output price/signal should remain positive
    assert out > 0


def test_signal_generator_uniform_range():
    # For uniform multiplicative noise with sigma = 0.2,
    # the multiplier should be in [0.8, 1.2]
    S_next = 100.0
    sigma = 0.2
    out = signal_generator(sigma, S_next, noise_distribution="uniform")

    # Check the resulting signal lies in the expected range
    assert 80.0 <= out <= 120.0


def test_signal_generator_uniform_large_sigma_positive():
    # Large sigma should still not produce a negative result because
    # the lower multiplier bound is clipped to stay positive
    out = signal_generator(2.0, 100.0, noise_distribution="uniform")

    # Output should still be positive
    assert out > 0


def test_signal_generator_invalid_distribution_raises():
    # Passing an invalid noise distribution should raise a ValueError
    with pytest.raises(ValueError):
        signal_generator(0.2, 100.0, noise_distribution="bad_dist")


def test_signal_generator_reproducible_with_seed():
    # Fix random seed for reproducibility
    np.random.seed(42)
    x1 = signal_generator(0.2, 100.0, noise_distribution="lognormal")

    # Reset seed and generate again
    np.random.seed(42)
    x2 = signal_generator(0.2, 100.0, noise_distribution="lognormal")

    # Outputs should match because the same seed was used
    assert np.isclose(x1, x2)


def test_signal_generator_uniform_zero_noise():
    # With zero uniform noise, multiplier should be exactly 1
    out = signal_generator(0.0, 100.0, noise_distribution="uniform")

    # Signal should remain unchanged
    assert out == 100.0


def test_signal_generator_lognormal_zero_noise_zero_bias():
    # With zero lognormal variance and zero bias, exp(0) = 1
    out = signal_generator(0.0, 100.0, bias=0.0, noise_distribution="lognormal")

    # Signal should remain unchanged
    assert out == 100.0


def test_signal_generator_lognormal_zero_noise_with_bias():
    # With zero variance, the normal draw collapses to the bias term,
    # so the result should be S_next * exp(bias)
    out = signal_generator(0.0, 100.0, bias=0.1, noise_distribution="lognormal")

    # Check against the expected deterministic value
    assert np.isclose(out, 100.0 * np.exp(0.1))


def test_assign_noise_bimodal_odd_num_agents():
    # Edge case: odd number of agents should still return the full count
    arr = assign_noise_parameter_set(7, "bimodal")

    # This test may fail with the current implementation if the bimodal
    # branch splits agents using floor division for both groups
    assert len(arr) == 7