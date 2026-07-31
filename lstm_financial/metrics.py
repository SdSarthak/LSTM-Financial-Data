"""Forecast quality metrics."""

from __future__ import annotations

import numpy as np


def _aligned(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype="float64").reshape(-1)
    predicted = np.asarray(predicted, dtype="float64").reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: actual {actual.shape} vs predicted {predicted.shape}")
    if actual.size == 0:
        raise ValueError("cannot score empty arrays")
    return actual, predicted


def mean_squared_error(actual, predicted) -> float:
    actual, predicted = _aligned(actual, predicted)
    return float(np.mean((actual - predicted) ** 2))


def root_mean_squared_error(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def mean_absolute_error(actual, predicted) -> float:
    actual, predicted = _aligned(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def mean_absolute_percentage_error(actual, predicted) -> float:
    """MAPE in percent, ignoring observations whose actual value is zero."""
    actual, predicted = _aligned(actual, predicted)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    errors = np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])
    return float(np.mean(errors) * 100.0)


def r_squared(actual, predicted) -> float:
    actual, predicted = _aligned(actual, predicted)
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    residual = float(np.sum((actual - predicted) ** 2))
    return 1.0 - residual / total


def directional_accuracy(previous, actual, predicted) -> float:
    """Share of steps where the predicted move has the same sign as the real one.

    ``previous`` is the last observed value before each forecast, so a flat
    prediction is only credited when the market was flat too.
    """
    previous = np.asarray(previous, dtype="float64").reshape(-1)
    actual, predicted = _aligned(actual, predicted)
    if previous.shape != actual.shape:
        raise ValueError("previous must align with actual")
    actual_move = np.sign(actual - previous)
    predicted_move = np.sign(predicted - previous)
    return float(np.mean(actual_move == predicted_move))


def regression_metrics(actual, predicted, previous=None) -> dict[str, float]:
    """Standard error metrics, plus directional accuracy when ``previous`` is given."""
    scores = {
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "MAPE": mean_absolute_percentage_error(actual, predicted),
        "R2": r_squared(actual, predicted),
    }
    if previous is not None:
        scores["DirectionalAccuracy"] = directional_accuracy(previous, actual, predicted)
    return scores


def format_metrics(scores: dict[str, float]) -> str:
    """Render a metric dict as aligned ``name: value`` lines."""
    if not scores:
        return "(no metrics)"
    width = max(len(name) for name in scores)
    return "\n".join(f"  {name:<{width}}  {value:.4f}" for name, value in scores.items())
