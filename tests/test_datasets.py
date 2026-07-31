import numpy as np
import pandas as pd
import pytest

from lstm_financial import datasets


def test_snake_case_normalises_source_columns():
    assert datasets.snake_case("Stock Splits") == "stock_splits"
    assert datasets.snake_case("Brand_Name") == "brand_name"
    assert datasets.snake_case("1.1 GVA at Basic Prices") == "1_1_gva_at_basic_prices"


def test_to_numeric_maps_missing_markers_to_nan():
    series = pd.Series(["1.5", "..", "", "2,000", "n.a."])
    converted = datasets.to_numeric(series)
    assert converted.iloc[0] == pytest.approx(1.5)
    assert converted.iloc[3] == pytest.approx(2000.0)
    assert converted.isna().sum() == 3


def test_load_stock_prices_normalises_and_sorts(stock_csv):
    frame = datasets.load_stock_prices(path=stock_csv, chunksize=5)
    assert {"date", "close", "ticker", "stock_splits"} <= set(frame.columns)
    assert frame["date"].dt.tz is None
    assert frame["ticker"].is_monotonic_increasing
    assert frame.groupby("ticker")["date"].is_monotonic_increasing.all()
    assert pd.api.types.is_float_dtype(frame["close"])


def test_load_stock_prices_filters_by_ticker(stock_csv):
    frame = datasets.load_stock_prices(path=stock_csv, tickers=["bbb"], chunksize=5)
    assert set(frame["ticker"]) == {"BBB"}
    assert len(frame) > 0


def test_load_stock_prices_returns_empty_frame_for_unknown_ticker(stock_csv):
    frame = datasets.load_stock_prices(path=stock_csv, tickers=["ZZZ"])
    assert frame.empty
    assert "close" in frame.columns


def test_load_ticker_history_deduplicates_and_indexes_by_date(stock_csv):
    history = datasets.load_ticker_history("BBB", path=stock_csv)
    assert history.index.name == "date"
    assert history.index.is_unique
    assert history.index.is_monotonic_increasing
    # The duplicated snapshot row wins for the final date.
    assert history["close"].iloc[-1] == pytest.approx(999.0)


def test_load_ticker_history_raises_for_missing_ticker(stock_csv):
    with pytest.raises(ValueError, match="No rows found"):
        datasets.load_ticker_history("ZZZ", path=stock_csv)


def test_list_tickers(stock_csv):
    assert datasets.list_tickers(path=stock_csv) == ["AAA", "BBB"]


def test_missing_file_raises_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download it first"):
        datasets.load_stock_prices(path=tmp_path / "nope.csv")


@pytest.mark.parametrize(
    ("label", "expected_month", "expected_frequency"),
    [
        ("2023 [2023]", 1, "A"),
        ("2023M04 [2023M04]", 4, "M"),
        ("2023Q2 [2023Q2]", 4, "Q"),
    ],
)
def test_parse_period(label, expected_month, expected_frequency):
    date, frequency = datasets.parse_period(label)
    assert date.year == 2023
    assert date.month == expected_month
    assert frequency == expected_frequency


def test_parse_period_rejects_non_periods():
    assert datasets.parse_period("Country Code") == (None, None)
    assert datasets.parse_period("2023M13") == (None, None)


def test_load_economic_indicators_melts_and_drops_missing(economic_csv):
    long = datasets.load_economic_indicators(path=economic_csv)
    assert set(long.columns) >= {"country_code", "series_code", "period", "frequency", "date", "value"}
    assert not long["value"].isna().any()
    assert set(long["frequency"]) == {"A", "M", "Q"}
    assert "[" not in "".join(long["period"])


def test_load_economic_indicators_filters_frequency(economic_csv):
    monthly = datasets.load_economic_indicators(path=economic_csv, frequency="M")
    assert set(monthly["frequency"]) == {"M"}
    assert len(monthly) == 4


def test_economic_series_returns_a_sorted_series(economic_csv):
    series = datasets.economic_series("DSTKMKTXN", "GBR", frequency="M", path=economic_csv)
    assert series.index.is_monotonic_increasing
    assert series.iloc[0] == pytest.approx(153.1)
    assert series.name == "DSTKMKTXN"


def test_economic_series_raises_for_unknown_code(economic_csv):
    with pytest.raises(ValueError, match="No observations"):
        datasets.economic_series("NOPE", path=economic_csv)


def test_rbi_quarterly_sheet_is_parsed(rbi_workbook):
    frame = datasets.load_rbi_indicators(path=rbi_workbook, sheet="Quarterly")
    assert list(frame.columns)[:2] == ["period", "gva_at_basic_prices"]
    assert frame["period"].tolist() == ["2024-25 Q4", "2024-25 Q3", "2024-25 Q2", "2023-24 Q4"]
    assert frame["gva_at_basic_prices"].iloc[0] == pytest.approx(6.7)
    # '..' becomes NaN rather than a string.
    assert np.isnan(frame["deposits"].iloc[2])
    assert frame.attrs["indicator_paths"]["agriculture"].startswith("1 Real Sector")


def test_rbi_monthly_sheet_is_parsed(rbi_workbook):
    frame = datasets.load_rbi_indicators(path=rbi_workbook, sheet="Monthly")
    assert "date" in frame.columns
    assert frame["date"].iloc[0] == pd.Timestamp("2025-04-30")
    assert frame["index_of_industrial_production"].iloc[0] == pytest.approx(2.5)
    assert {"deposits", "credit"} <= set(frame.columns)
    # The stray '2025' grouping row is not data.
    assert len(frame) == 3


def test_rbi_column_names_do_not_mix_sections(rbi_workbook):
    frame = datasets.load_rbi_indicators(path=rbi_workbook, sheet="Monthly")
    assert frame.attrs["indicator_paths"]["credit"].endswith("2.1.2 Credit")
    assert "Real Sector" not in frame.attrs["indicator_paths"]["credit"]


def test_synthetic_series_is_deterministic():
    first = datasets.synthetic_price_series(periods=50, seed=5)
    second = datasets.synthetic_price_series(periods=50, seed=5)
    pd.testing.assert_series_equal(first, second)
    assert not first.equals(datasets.synthetic_price_series(periods=50, seed=6))


def test_synthetic_ohlcv_is_internally_consistent():
    frame = datasets.synthetic_ohlcv(periods=40, seed=2)
    assert (frame["high"] >= frame["close"]).all()
    assert (frame["low"] <= frame["close"]).all()
    assert (frame["high"] >= frame["low"]).all()
    assert frame.index.is_monotonic_increasing
    assert frame["volume"].gt(0).all()
