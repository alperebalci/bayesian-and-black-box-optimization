"""Deterministic regression tests for maintained, native Bayesian optimization methods."""

import numpy as np
import pytest

from src.discrete_bo import DiscreteGaussianProcessBayesOptimizer
from src.gaussian_bo import GaussianProcessBayesOptimizer
from src.multi_fidelity import TwoFidelitySurrogate


def test_continuous_bounds_and_nonfinite_objective_validation():
    with pytest.raises(ValueError):
        GaussianProcessBayesOptimizer(lambda x: float(x[0]), bounds=[[1.0, 0.0]])
    optimizer = GaussianProcessBayesOptimizer(
        lambda x: float("nan"), bounds=[[0.0, 1.0]],
        n_initial_points=3, n_restarts_optimizer=0, random_state=7,
    )
    with pytest.raises(ValueError, match="finite"):
        optimizer.optimize(n_iterations=0)


def test_continuous_bayesian_cycle_is_feasible_and_reproducible():
    def run(seed):
        opt = GaussianProcessBayesOptimizer(
            lambda x: float((x[0] - 0.3) ** 2),
            bounds=[[0.0, 1.0]],
            n_initial_points=3,
            n_restarts_optimizer=0,
            random_state=seed,
        )
        # Keep the CI evaluation small while exercising actual GP fitting.
        opt._select_next_point = lambda: np.array([0.3])
        return opt.optimize(n_iterations=1)

    left = run(14)
    right = run(14)
    assert len(left.y_observed) == 4
    assert len(left.history) == 1
    assert np.all((left.X_observed >= 0.0) & (left.X_observed <= 1.0))
    assert left.best_y == pytest.approx(0.0)
    assert np.allclose(left.X_observed, right.X_observed)
    assert np.allclose(left.y_observed, right.y_observed)


def test_discrete_bayesian_optimizer_does_not_reevaluate_candidates():
    candidates = np.arange(9, dtype=float).reshape(-1, 1)
    opt = DiscreteGaussianProcessBayesOptimizer(
        lambda x: float((x[0] - 4.0) ** 2),
        candidate_points=candidates,
        n_initial_points=3,
        n_restarts_optimizer=0,
        random_state=12,
    )
    result = opt.optimize(n_iterations=3)
    assert len(result.observed_indices) == 6
    assert len(set(result.observed_indices)) == 6
    assert result.best_y == pytest.approx(min(result.y_observed))
    assert all(0 <= i < len(candidates) for i in result.observed_indices)


def test_two_fidelity_predictions_remain_finite():
    surrogate = TwoFidelitySurrogate(n_dimensions=1, random_state=9)
    low_x = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
    high_x = np.array([[0.0], [0.5], [1.0]])
    low_y = low_x[:, 0] ** 2
    high_y = 1.2 * high_x[:, 0] ** 2 + 0.05
    with pytest.raises(RuntimeError):
        surrogate.predict_high(high_x)
    surrogate.fit(low_x, low_y, high_x, high_y)
    mean, std = surrogate.predict_high(np.array([[0.25], [0.75]]))
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0.0)
