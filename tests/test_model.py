"""Model tests.

They train for a couple of epochs on a handful of synthetic windows: the point
is that the wiring works and shapes line up, not that the network is accurate.
"""

import numpy as np
import pytest

from lstm_financial import features, windows
from lstm_financial.model import FinancialLSTM, LSTMConfig, persistence_forecast, set_seed

tensorflow = pytest.importorskip("tensorflow", reason="TensorFlow is optional")


@pytest.fixture(scope="module")
def small_dataset():
    from lstm_financial import datasets

    frame = datasets.synthetic_price_series(periods=160, seed=4).to_frame("close")
    return windows.build_dataset(frame, lookback=8, test_fraction=0.2, validation_fraction=0.1)


@pytest.fixture(scope="module")
def trained(small_dataset):
    model = FinancialLSTM.from_dataset(small_dataset, units=(8,), epochs=2, patience=0, seed=1)
    history = model.fit(
        small_dataset.x_train,
        small_dataset.y_train,
        validation_data=(small_dataset.x_validation, small_dataset.y_validation),
        verbose=0,
    )
    return model, history


def test_config_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError):
        LSTMConfig(lookback=0)
    with pytest.raises(ValueError):
        LSTMConfig(units=())
    with pytest.raises(ValueError):
        LSTMConfig(dropout=1.0)


def test_from_dataset_matches_the_dataset_shape(small_dataset):
    model = FinancialLSTM.from_dataset(small_dataset)
    assert model.config.lookback == small_dataset.lookback
    assert model.config.n_features == small_dataset.n_features
    assert model.config.horizon == small_dataset.horizon


def test_build_produces_the_expected_input_and_output_shape(small_dataset):
    model = FinancialLSTM.from_dataset(small_dataset, units=(8, 4))
    keras_model = model.build()
    assert keras_model.input_shape == (None, small_dataset.lookback, small_dataset.n_features)
    assert keras_model.output_shape == (None, small_dataset.horizon)
    assert any("lstm" in layer.name for layer in keras_model.layers)


def test_predict_before_fit_is_an_error(small_dataset):
    with pytest.raises(RuntimeError, match="not built or trained"):
        FinancialLSTM.from_dataset(small_dataset).predict(small_dataset.x_test)


def test_fit_returns_history_and_predicts_the_right_shape(trained, small_dataset):
    model, history = trained
    assert "loss" in history and len(history["loss"]) == 2
    assert "val_loss" in history
    predictions = model.predict(small_dataset.x_test)
    assert predictions.shape == (len(small_dataset.x_test), small_dataset.horizon)
    assert np.isfinite(predictions).all()


def test_forecast_extends_the_series_by_the_requested_steps(trained, small_dataset):
    model, _ = trained
    forecast = model.forecast(small_dataset.last_window(), steps=5)
    assert forecast.shape == (5,)
    assert np.isfinite(forecast).all()


def test_forecast_rejects_multivariate_windows(price_frame):
    enriched = features.select_features(features.add_technical_indicators(price_frame))
    dataset = windows.build_dataset(enriched, lookback=8)
    model = FinancialLSTM.from_dataset(dataset, units=(4,))
    model.build()
    with pytest.raises(ValueError, match="univariate"):
        model.forecast(dataset.last_window(), steps=3)


def test_save_and_load_round_trip(trained, small_dataset, tmp_path):
    model, _ = trained
    path = model.save(tmp_path / "model")
    assert path.suffix == ".keras"
    reloaded = FinancialLSTM.load(path)
    assert reloaded.config.lookback == small_dataset.lookback
    np.testing.assert_allclose(
        reloaded.predict(small_dataset.x_test),
        model.predict(small_dataset.x_test),
        rtol=1e-5,
        atol=1e-6,
    )


def test_seeding_makes_initial_weights_reproducible(small_dataset):
    first = FinancialLSTM.from_dataset(small_dataset, units=(4,), seed=123)
    second = FinancialLSTM.from_dataset(small_dataset, units=(4,), seed=123)
    first.build()
    second.build()
    for left, right in zip(first.model.get_weights(), second.model.get_weights()):
        np.testing.assert_allclose(left, right)


def test_set_seed_makes_numpy_draws_repeatable():
    set_seed(7)
    first = np.random.rand(5)
    set_seed(7)
    np.testing.assert_array_equal(first, np.random.rand(5))


def test_persistence_forecast_repeats_the_last_value():
    baseline = persistence_forecast(np.array([1.0, 2.0]), horizon=3)
    assert baseline.shape == (2, 3)
    np.testing.assert_array_equal(baseline[0], [1.0, 1.0, 1.0])
