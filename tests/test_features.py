import numpy as np
import pandas as pd
import pytest

from lstm_financial import features


def test_simple_moving_average_matches_manual_mean(linear_series):
    sma = features.simple_moving_average(linear_series, 3)
    assert np.isnan(sma.iloc[0]) and np.isnan(sma.iloc[1])
    assert sma.iloc[2] == pytest.approx(2.0)
    assert sma.iloc[-1] == pytest.approx((58 + 59 + 60) / 3)
    assert sma.name == "sma_3"


def test_exponential_moving_average_recursion():
    prices = pd.Series([10.0, 20.0, 30.0])
    ema = features.exponential_moving_average(prices, span=3)
    alpha = 2 / (3 + 1)
    expected_second = 10.0 + alpha * (20.0 - 10.0)
    assert ema.iloc[0] == pytest.approx(10.0)
    assert ema.iloc[1] == pytest.approx(expected_second)
    assert ema.iloc[2] == pytest.approx(expected_second + alpha * (30.0 - expected_second))


def test_log_returns_first_value_is_nan_and_rest_match_numpy(linear_series):
    returns = features.log_returns(linear_series)
    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(np.log(2.0) - np.log(1.0))


def test_rsi_saturates_at_100_for_a_rising_series(linear_series):
    rsi = features.relative_strength_index(linear_series, period=14)
    assert np.isnan(rsi.iloc[:14]).all()
    assert rsi.dropna().eq(100.0).all()


def test_rsi_is_zero_for_a_falling_series(linear_series):
    rsi = features.relative_strength_index(linear_series.iloc[::-1].reset_index(drop=True), period=14)
    assert rsi.dropna().eq(0.0).all()


def test_rsi_is_neutral_for_a_flat_series():
    flat = pd.Series([50.0] * 30)
    rsi = features.relative_strength_index(flat, period=14)
    assert rsi.dropna().eq(50.0).all()


def test_rsi_stays_in_range_for_a_noisy_series(price_frame):
    rsi = features.relative_strength_index(price_frame["close"], period=14).dropna()
    assert len(rsi) > 0
    assert rsi.between(0.0, 100.0).all()


def test_macd_histogram_is_line_minus_signal(price_frame):
    frame = features.macd(price_frame["close"])
    assert list(frame.columns) == ["macd", "macd_signal", "macd_histogram"]
    difference = frame["macd"] - frame["macd_signal"]
    pd.testing.assert_series_equal(frame["macd_histogram"], difference, check_names=False)


def test_macd_rejects_fast_slower_than_slow(price_frame):
    with pytest.raises(ValueError):
        features.macd(price_frame["close"], fast=26, slow=12)


def test_bollinger_percent_b_is_half_at_the_middle_band(price_frame):
    close = price_frame["close"]
    bands = features.bollinger_bands(close, window=20).dropna()
    assert (bands["bb_upper"] >= bands["bb_middle"]).all()
    assert (bands["bb_lower"] <= bands["bb_middle"]).all()

    # %B is 0 at the lower band, 0.5 at the middle band and 1 at the upper band.
    reconstructed = (close.loc[bands.index] - bands["bb_lower"]) / (
        bands["bb_upper"] - bands["bb_lower"]
    )
    pd.testing.assert_series_equal(bands["bb_percent_b"], reconstructed, check_names=False)
    at_middle = np.isclose(close.loc[bands.index], bands["bb_middle"])
    assert bands.loc[at_middle, "bb_percent_b"].sub(0.5).abs().lt(1e-9).all()


def test_realised_volatility_is_zero_for_a_constant_series():
    flat = pd.Series([100.0] * 60, index=pd.bdate_range("2024-01-01", periods=60))
    volatility = features.realised_volatility(flat, window=20).dropna()
    assert volatility.eq(0.0).all()


def test_add_technical_indicators_drops_warmup_and_adds_columns(price_frame):
    enriched = features.add_technical_indicators(price_frame)
    for column in features.DEFAULT_FEATURE_COLUMNS:
        assert column in enriched.columns
    assert not enriched.isna().to_numpy().any()
    assert len(enriched) < len(price_frame)
    assert enriched.index.is_monotonic_increasing


def test_add_technical_indicators_keeps_rows_when_a_source_column_is_empty(price_frame):
    frame = price_frame.copy()
    frame["capital_gains"] = np.nan  # exactly how the real stock file ships it
    enriched = features.add_technical_indicators(frame)
    assert len(enriched) == len(features.add_technical_indicators(price_frame))
    assert enriched["capital_gains"].isna().all()


def test_add_technical_indicators_requires_the_price_column(price_frame):
    with pytest.raises(KeyError):
        features.add_technical_indicators(price_frame, price_column="adj_close")


def test_select_features_reports_missing_columns(price_frame):
    enriched = features.add_technical_indicators(price_frame)
    with pytest.raises(KeyError, match="Missing feature columns"):
        features.select_features(enriched, ["close", "not_a_feature"])
    selected = features.select_features(enriched, ["close", "rsi_14"])
    assert list(selected.columns) == ["close", "rsi_14"]
