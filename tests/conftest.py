"""Shared synthetic fixtures - the tests never read the real datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstm_financial import datasets


@pytest.fixture(scope="session")
def price_frame() -> pd.DataFrame:
    """Deterministic OHLCV frame with enough history for a 20-step lookback."""
    return datasets.synthetic_ohlcv(periods=400, seed=11, ticker="TEST")


@pytest.fixture
def stock_csv(tmp_path):
    """A miniature copy of the world stock prices file layout."""
    dates = pd.bdate_range("2024-01-01", periods=12)
    rows = []
    for offset, ticker in enumerate(["AAA", "BBB"]):
        for index, date in enumerate(dates):
            close = 100.0 + offset * 50 + index
            rows.append(
                {
                    "Date": f"{date:%Y-%m-%d} 00:00:00-05:00",
                    "Open": close - 0.5,
                    "High": close + 1.0,
                    "Low": close - 1.0,
                    "Close": close,
                    "Volume": 1_000_000 + index,
                    "Brand_Name": ticker.lower(),
                    "Ticker": ticker,
                    "Industry_Tag": "test",
                    "Country": "usa",
                    "Dividends": 0.0,
                    "Stock Splits": 0.0,
                    "Capital Gains": "",
                }
            )
    # A duplicated final row, exactly like the appended snapshots in the source file.
    rows.append(dict(rows[-1], Close=999.0))
    path = tmp_path / "stocks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def economic_csv(tmp_path):
    """Wide `series x period` export with the '..' missing marker."""
    frame = pd.DataFrame(
        [
            {
                "Country": "United Kingdom",
                "Country Code": "GBR",
                "Series": "Stock Markets, LCU,,,",
                "Series Code": "DSTKMKTXN",
                "2023 [2023]": "150.0",
                "2023M01 [2023M01]": "153.1",
                "2023M02 [2023M02]": "156.4",
                "2023Q1 [2023Q1]": "..",
            },
            {
                "Country": "India",
                "Country Code": "IND",
                "Series": "Core CPI,not seas.adj,,,",
                "Series Code": "CORENS",
                "2023 [2023]": "..",
                "2023M01 [2023M01]": "138.8",
                "2023M02 [2023M02]": "140.5",
                "2023Q1 [2023Q1]": "139.6",
            },
        ]
    )
    path = tmp_path / "economic.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def rbi_workbook(tmp_path):
    """Workbook mirroring the RBI layout: title rows, merged headers, sparse years."""
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()

    quarterly = workbook.active
    quarterly.title = "Quarterly"
    quarterly["B2"] = "Select Economic Indicators (Test)"
    quarterly["B4"] = " Year"
    quarterly["C4"] = " Quarter"
    quarterly["D4"] = "1 Real Sector (% Change)"
    quarterly["D5"] = "1.1 GVA at Basic Prices"
    quarterly["E5"] = "1.1.1 Agriculture"
    quarterly["F4"] = "2 Money and Banking (% Change)"
    quarterly["F5"] = "2.1 Deposits"
    values = [
        ("2024-25", "Q4", 6.7, 5.3, 10.1),
        (None, "Q3", 6.4, 6.6, 10.4),
        (None, "Q2", 5.8, 4.1, ".."),
        ("2023-24", "Q4", 7.2, 0.8, 12.0),
    ]
    for offset, row in enumerate(values):
        for column, value in enumerate(row):
            if value is not None:
                quarterly.cell(row=6 + offset, column=2 + column, value=value)

    monthly = workbook.create_sheet("Monthly")
    monthly["B2"] = "Select Economic Indicators (Test)"
    monthly["C4"] = "1 Real Sector (% Change)"
    monthly["D4"] = "2 Money and Banking (% Change)"
    monthly["B5"] = "Date"
    monthly["C5"] = "1.2 Index of Industrial Production"
    monthly["D5"] = "2.1 Scheduled Commercial Banks"
    monthly["D6"] = "2.1.1 Deposits"
    monthly["E6"] = "2.1.2 Credit"
    monthly.cell(row=7, column=2, value="2025")
    monthly_rows = [
        (pd.Timestamp("2025-04-30"), 2.5, 10.0, 11.0),
        (pd.Timestamp("2025-03-31"), 3.9, 10.5, 12.0),
        (pd.Timestamp("2025-02-28"), 1.1, "..", 12.5),
    ]
    for offset, row in enumerate(monthly_rows):
        for column, value in enumerate(row):
            monthly.cell(row=8 + offset, column=2 + column, value=value)

    path = tmp_path / "rbi.xlsx"
    workbook.save(path)
    return path


@pytest.fixture
def linear_series() -> pd.Series:
    """Strictly increasing series - handy for closed-form indicator checks."""
    index = pd.bdate_range("2024-01-01", periods=60)
    return pd.Series(np.arange(1, 61, dtype="float64"), index=index, name="close")
