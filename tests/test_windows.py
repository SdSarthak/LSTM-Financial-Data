import numpy as np
import pandas as pd
import pytest

from lstm_financial import features, windows


def test_make_sequences_shapes_and_alignment():
    values = np.arange(10, dtype="float64").reshape(-1, 1)
    x, y = windows.make_sequences(values, values.reshape(-1), lookback=3, horizon=1)
    assert x.shape == (7, 3, 1)
    assert y.shape == (7, 1)
    np.testing.assert_array_equal(x[0].reshape(-1), [0, 1, 2])
    np.testing.assert_array_equal(y[0], [3])
    np.testing.assert_array_equal(x[-1].reshape(-1), [6, 7, 8])
    np.testing.assert_array_equal(y[-1], [9])


def test_make_sequences_multi_step_horizon():
    values = np.arange(10, dtype="float64")
    x, y = windows.make_sequences(values, values, lookback=3, horizon=2)
    assert x.shape == (6, 3, 1)
    np.testing.assert_array_equal(y[0], [3, 4])
    np.testing.assert_array_equal(y[-1], [8, 9])


def test_make_sequences_rejects_short_input():
    with pytest.raises(ValueError, match="at least"):
        windows.make_sequences(np.arange(4.0), np.arange(4.0), lookback=5)


def test_make_sequences_rejects_misaligned_targets():
    with pytest.raises(ValueError, match="same number of rows"):
        windows.make_sequences(np.arange(10.0), np.arange(9.0), lookback=3)


def test_chronological_split_is_ordered_and_covers_every_row():
    train, validation, test = windows.chronological_split(100, 0.2, 0.1)
    assert (train.start, train.stop) == (0, 70)
    assert (validation.start, validation.stop) == (70, 80)
    assert (test.start, test.stop) == (80, 100)


def test_chronological_split_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        windows.chronological_split(100, 0.8, 0.3)


def _dataset(price_frame, **kwargs):
    enriched = features.add_technical_indicators(price_frame)
    frame = features.select_features(enriched, ["close", "rsi_14", "log_return"])
    return frame, windows.build_dataset(frame, lookback=10, **kwargs)


def test_build_dataset_splits_are_disjoint_and_shaped(price_frame):
    frame, dataset = _dataset(price_frame)
    assert dataset.n_features == 3
    assert dataset.x_train.shape[1:] == (10, 3)
    assert len(dataset.x_train) > len(dataset.x_test) > 0
    assert dataset.has_validation
    assert len(dataset.test_index) == len(dataset.x_test)
    assert dataset.test_index.max() == frame.index.max()


def test_scalers_are_fitted_on_training_rows_only(price_frame):
    frame, dataset = _dataset(price_frame)
    train_rows, _, _ = windows.chronological_split(len(frame), 0.2, 0.1)
    train_close = frame["close"].to_numpy()[train_rows]
    restored_min = dataset.target_scaler.inverse_transform([[0.0]])[0][0]
    restored_max = dataset.target_scaler.inverse_transform([[1.0]])[0][0]
    assert restored_min == pytest.approx(train_close.min())
    assert restored_max == pytest.approx(train_close.max())

    # Rows after the training block may legitimately scale outside [0, 1].
    scaled_test_targets = dataset.y_test
    assert scaled_test_targets.size > 0
    assert dataset.x_train.min() >= -1e-9
    assert dataset.x_train.max() <= 1.0 + 1e-9


def test_inverse_target_round_trips(price_frame):
    frame, dataset = _dataset(price_frame)
    actual = dataset.inverse_target(dataset.y_test).reshape(-1)
    expected = frame["close"].to_numpy()[-len(actual) :]
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_previous_actual_is_the_value_before_each_target(price_frame):
    frame, dataset = _dataset(price_frame)
    previous = dataset.previous_actual("test")
    actual = dataset.inverse_target(dataset.y_test)[:, 0]
    close = frame["close"].to_numpy()
    positions = [np.argmin(np.abs(close - value)) for value in actual]
    expected = close[[p - 1 for p in positions]]
    np.testing.assert_allclose(previous, expected, rtol=1e-9)


def test_build_dataset_rejects_nans():
    frame = pd.DataFrame({"close": [1.0, np.nan, 3.0, 4.0, 5.0]})
    with pytest.raises(ValueError, match="NaN"):
        windows.build_dataset(frame, lookback=2)


def test_build_dataset_requires_a_known_target():
    frame = pd.DataFrame({"close": np.arange(50.0)})
    with pytest.raises(KeyError):
        windows.build_dataset(frame, target_column="price", lookback=5)


def test_last_window_has_model_input_shape(price_frame):
    _, dataset = _dataset(price_frame)
    assert dataset.last_window().shape == (10, 3)
