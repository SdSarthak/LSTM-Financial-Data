import numpy as np
import pytest

from lstm_financial import metrics


def test_perfect_predictions_score_zero_error():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    scores = metrics.regression_metrics(actual, actual)
    assert scores["MSE"] == pytest.approx(0.0)
    assert scores["RMSE"] == pytest.approx(0.0)
    assert scores["MAE"] == pytest.approx(0.0)
    assert scores["MAPE"] == pytest.approx(0.0)
    assert scores["R2"] == pytest.approx(1.0)


def test_known_error_values():
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0, 18.0])
    assert metrics.mean_squared_error(actual, predicted) == pytest.approx(4.0)
    assert metrics.root_mean_squared_error(actual, predicted) == pytest.approx(2.0)
    assert metrics.mean_absolute_error(actual, predicted) == pytest.approx(2.0)
    assert metrics.mean_absolute_percentage_error(actual, predicted) == pytest.approx(15.0)


def test_mape_ignores_zero_denominators():
    actual = np.array([0.0, 10.0])
    predicted = np.array([5.0, 11.0])
    assert metrics.mean_absolute_percentage_error(actual, predicted) == pytest.approx(10.0)


def test_mape_is_nan_when_every_actual_is_zero():
    assert np.isnan(metrics.mean_absolute_percentage_error([0.0, 0.0], [1.0, 2.0]))


def test_r_squared_is_nan_for_a_constant_target():
    assert np.isnan(metrics.r_squared([5.0, 5.0, 5.0], [4.0, 5.0, 6.0]))


def test_r_squared_of_the_mean_predictor_is_zero():
    actual = np.array([1.0, 2.0, 3.0])
    assert metrics.r_squared(actual, np.full(3, actual.mean())) == pytest.approx(0.0)


def test_directional_accuracy_counts_matching_moves():
    previous = np.array([10.0, 10.0, 10.0, 10.0])
    actual = np.array([11.0, 9.0, 11.0, 9.0])
    predicted = np.array([12.0, 8.0, 9.0, 8.0])
    assert metrics.directional_accuracy(previous, actual, predicted) == pytest.approx(0.75)


def test_flat_forecasts_never_call_a_move():
    previous = np.array([10.0, 10.0])
    actual = np.array([11.0, 9.0])
    assert metrics.directional_accuracy(previous, actual, previous) == pytest.approx(0.0)


def test_shape_and_emptiness_are_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.mean_absolute_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        metrics.mean_absolute_error([], [])


def test_regression_metrics_adds_direction_only_when_previous_given():
    actual = [1.0, 2.0]
    predicted = [1.1, 1.9]
    assert "DirectionalAccuracy" not in metrics.regression_metrics(actual, predicted)
    with_previous = metrics.regression_metrics(actual, predicted, previous=[0.9, 2.1])
    assert "DirectionalAccuracy" in with_previous


def test_format_metrics_renders_aligned_lines():
    text = metrics.format_metrics({"RMSE": 1.5, "MAE": 1.0})
    assert "RMSE" in text and "1.5000" in text
    assert len(text.splitlines()) == 2
    assert metrics.format_metrics({}) == "(no metrics)"
