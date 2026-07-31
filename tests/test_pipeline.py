import json

import numpy as np
import pytest

from lstm_financial import datasets, pipeline
from lstm_financial.features import DEFAULT_FEATURE_COLUMNS, DEFAULT_RETURN_FEATURE_COLUMNS

pytest.importorskip("tensorflow", reason="TensorFlow is optional")


@pytest.fixture(scope="module")
def univariate_run(tmp_path_factory):
    frame = datasets.synthetic_ohlcv(periods=220, seed=8)
    return pipeline.run_forecast(
        frame,
        name="test",
        feature_columns=("log_return",),
        lookback=8,
        epochs=2,
        patience=0,
        forecast_steps=4,
        artifacts_dir=tmp_path_factory.mktemp("artifacts"),
        verbose=0,
    )


def test_prepare_features_returns_the_default_columns(price_frame):
    prepared = pipeline.prepare_features(price_frame)
    assert list(prepared.columns) == list(DEFAULT_FEATURE_COLUMNS)
    assert not prepared.isna().to_numpy().any()


def test_prepare_features_can_keep_every_numeric_column(price_frame):
    prepared = pipeline.prepare_features(price_frame, feature_columns=None)
    assert "bb_upper" in prepared.columns
    assert "ticker" not in prepared.columns


def test_enrich_adds_log_returns_without_indicators(price_frame):
    enriched = pipeline.enrich(price_frame[["close"]], add_indicators=False)
    assert list(enriched.columns) == ["close", "log_return"]
    assert len(enriched) == len(price_frame) - 1


def test_run_defaults_to_return_mode(univariate_run):
    assert univariate_run.target_mode == "return"
    assert univariate_run.config["target_column"] == "log_return"
    assert univariate_run.config["features"] == ["log_return"]


def test_run_produces_metrics_a_baseline_and_a_forecast(univariate_run):
    assert set(univariate_run.metrics) == {"MSE", "RMSE", "MAE", "MAPE", "R2", "DirectionalAccuracy"}
    assert set(univariate_run.baseline_metrics) == set(univariate_run.metrics)
    assert all(np.isfinite(value) for value in univariate_run.metrics.values())
    assert univariate_run.future_forecast is not None
    assert univariate_run.future_forecast.shape == (4,)
    assert (univariate_run.future_forecast > 0).all()


def test_metrics_are_reported_in_price_units(univariate_run):
    # Reconstructed prices must sit in the same range as the synthetic series.
    assert univariate_run.actuals.min() > 50.0
    assert univariate_run.actuals.max() < 200.0
    assert univariate_run.metrics["MAPE"] < 100.0


def test_actuals_reconstruct_the_true_close_prices(univariate_run):
    frame = datasets.synthetic_ohlcv(periods=220, seed=8)
    enriched = pipeline.enrich(frame)
    expected = enriched["close"].loc[univariate_run.index].to_numpy()
    np.testing.assert_allclose(univariate_run.actuals[:, 0], expected, rtol=1e-9)


def test_run_predictions_align_with_actuals(univariate_run):
    assert univariate_run.predictions.shape == univariate_run.actuals.shape
    assert len(univariate_run.index) == len(univariate_run.actuals)
    assert univariate_run.index.is_monotonic_increasing


def test_run_writes_every_artifact(univariate_run):
    artifacts = univariate_run.artifacts
    assert {"predictions_plot", "history_plot", "forecast_plot", "model", "summary"} <= set(artifacts)
    summary = json.loads(open(artifacts["summary"], encoding="utf-8").read())
    assert summary["config"]["lookback"] == 8
    assert summary["target_mode"] == "return"
    assert summary["test_windows"] == len(univariate_run.actuals)
    assert summary["metrics"]["RMSE"] == pytest.approx(univariate_run.metrics["RMSE"])
    assert summary["beats_baseline"] == univariate_run.beats_baseline()


def test_report_mentions_both_models(univariate_run):
    report = univariate_run.report()
    assert "LSTM:" in report and "Persistence baseline" in report
    assert "Recursive forecast" in report


def test_level_mode_uses_the_price_as_target(tmp_path):
    frame = datasets.synthetic_ohlcv(periods=220, seed=12)
    run = pipeline.run_forecast(
        frame,
        name="level",
        target_mode="level",
        feature_columns=("close",),
        lookback=8,
        epochs=1,
        patience=0,
        forecast_steps=3,
        artifacts_dir=tmp_path,
        verbose=0,
    )
    assert run.target_mode == "level"
    assert run.config["target_column"] == "close"
    assert run.future_forecast.shape == (3,)


def test_multivariate_run_skips_the_recursive_forecast(tmp_path):
    frame = datasets.synthetic_ohlcv(periods=220, seed=9)
    run = pipeline.run_forecast(
        frame,
        name="multi",
        lookback=8,
        epochs=1,
        patience=0,
        artifacts_dir=tmp_path,
        verbose=0,
    )
    assert run.future_forecast is None
    assert "forecast_plot" not in run.artifacts
    assert run.config["features"] == list(DEFAULT_RETURN_FEATURE_COLUMNS)
    assert "Recursive forecast: skipped" in run.report()


def test_multi_step_horizon_predicts_a_price_path(tmp_path):
    frame = datasets.synthetic_ohlcv(periods=260, seed=13)
    run = pipeline.run_forecast(
        frame,
        name="horizon",
        feature_columns=("log_return", "rsi_14"),
        lookback=8,
        horizon=3,
        epochs=1,
        patience=0,
        artifacts_dir=tmp_path,
        verbose=0,
    )
    assert run.predictions.shape[1] == 3
    assert run.actuals.shape == run.predictions.shape


def test_unknown_target_mode_is_rejected(price_frame):
    with pytest.raises(ValueError, match="target_mode"):
        pipeline.run_forecast(price_frame, target_mode="prices", epochs=1, verbose=0)


def test_too_short_a_test_split_is_reported_clearly():
    frame = datasets.synthetic_ohlcv(periods=120, seed=10)
    with pytest.raises(ValueError, match="test split is too short"):
        pipeline.run_forecast(
            frame, feature_columns=("log_return",), lookback=60, epochs=1, verbose=0, test_fraction=0.05
        )
