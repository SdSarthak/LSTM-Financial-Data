"""Loaders for the datasets used by this project.

None of the raw data is committed to the repository (see ``README.md`` for
download instructions); every loader takes an explicit path or falls back to
the environment-driven :class:`~lstm_financial.config.Settings`.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import Settings, get_settings

# Values the source files use to mark "no observation".
MISSING_TOKENS = ("..", "...", "-", "--", "n.a.", "N.A.", "NA", "")

_PERIOD_RE = re.compile(r"^(?P<year>\d{4})(?:(?P<kind>[MQ])(?P<index>\d{1,2}))?$")


def _resolve(path: Path | str | None, fallback: Path) -> Path:
    resolved = Path(path) if path is not None else fallback
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}. Download it first (see README) or "
            f"point LSTM_FIN_DATA_DIR at the directory that holds it."
        )
    return resolved


def snake_case(name: str) -> str:
    """``"Stock Splits"`` -> ``"stock_splits"``."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
    cleaned = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cleaned)
    return cleaned.lower()


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column of strings to float, mapping missing markers to NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(list(MISSING_TOKENS), pd.NA)
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


# --------------------------------------------------------------------------
# World stock prices (Kaggle: World Stock Prices Daily Updating)
# --------------------------------------------------------------------------

STOCK_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "dividends", "stock_splits")


def _normalise_stock_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={c: snake_case(c) for c in frame.columns})
    if "date" in frame.columns:
        parsed = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["date"] = parsed.dt.tz_convert(None).dt.normalize()
        frame = frame.dropna(subset=["date"])
    for column in STOCK_NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = to_numeric(frame[column])
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    return frame


def load_stock_prices(
    path: Path | str | None = None,
    tickers: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
    chunksize: int = 250_000,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load the world stock prices CSV, optionally filtered to some tickers.

    The file is several tens of megabytes, so it is streamed in chunks and
    filtered on the way in rather than loaded whole.
    """
    settings = settings or get_settings()
    csv_path = _resolve(path, settings.stock_path)
    wanted = {t.strip().upper() for t in tickers} if tickers else None

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=columns, low_memory=False):
        chunk = _normalise_stock_frame(chunk)
        if wanted is not None:
            if "ticker" not in chunk.columns:
                raise KeyError("Cannot filter by ticker: the file has no Ticker column")
            chunk = chunk[chunk["ticker"].isin(wanted)]
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        return _normalise_stock_frame(pd.read_csv(csv_path, nrows=0, usecols=columns))

    frame = pd.concat(frames, ignore_index=True)
    sort_columns = [c for c in ("ticker", "date") if c in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return frame


def list_tickers(
    path: Path | str | None = None,
    chunksize: int = 250_000,
    settings: Settings | None = None,
) -> list[str]:
    """Return every ticker symbol present in the stock price file."""
    settings = settings or get_settings()
    csv_path = _resolve(path, settings.stock_path)
    seen: set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=["Ticker"], chunksize=chunksize):
        seen.update(chunk["Ticker"].dropna().astype(str).str.strip().str.upper())
    return sorted(seen)


def load_ticker_history(
    ticker: str,
    path: Path | str | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Return one ticker's daily history indexed by date, ascending and unique.

    Duplicate rows for the same day (the source file appends new snapshots) are
    collapsed keeping the last observation.
    """
    frame = load_stock_prices(path=path, tickers=[ticker], settings=settings)
    if frame.empty:
        raise ValueError(f"No rows found for ticker {ticker!r}")
    frame = (
        frame.drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
        .sort_index()
    )
    frame.index.name = "date"
    return frame


# --------------------------------------------------------------------------
# World Bank / IMF style wide "series x period" export
# --------------------------------------------------------------------------


def parse_period(label: str) -> tuple[pd.Timestamp, str] | tuple[None, None]:
    """Parse ``2023``/``2023M04``/``2023Q2`` into (period start, frequency)."""
    token = str(label).split("[")[0].strip()
    match = _PERIOD_RE.match(token)
    if not match:
        return None, None
    year = int(match.group("year"))
    kind = match.group("kind")
    if kind is None:
        return pd.Timestamp(year=year, month=1, day=1), "A"
    index = int(match.group("index"))
    if kind == "M":
        if not 1 <= index <= 12:
            return None, None
        return pd.Timestamp(year=year, month=index, day=1), "M"
    if not 1 <= index <= 4:
        return None, None
    return pd.Timestamp(year=year, month=3 * (index - 1) + 1, day=1), "Q"


def load_economic_indicators(
    path: Path | str | None = None,
    frequency: str | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load the wide economic indicator export into tidy long format.

    Returns columns ``country, country_code, series, series_code, period,
    frequency, date, value`` with missing markers dropped.  ``frequency`` may be
    ``"A"``, ``"Q"`` or ``"M"`` to keep a single periodicity.
    """
    settings = settings or get_settings()
    csv_path = _resolve(path, settings.economic_path)
    raw = pd.read_csv(csv_path, dtype=str)

    id_columns = [c for c in raw.columns if not parse_period(c)[0]]
    value_columns = [c for c in raw.columns if c not in id_columns]
    if not value_columns:
        raise ValueError(f"{csv_path.name} has no period columns to melt")

    long = raw.melt(id_vars=id_columns, value_vars=value_columns, var_name="period", value_name="value")
    long = long.rename(columns={c: snake_case(c) for c in id_columns})
    parsed = long["period"].map(parse_period)
    long["date"] = [p[0] for p in parsed]
    long["frequency"] = [p[1] for p in parsed]
    long["period"] = long["period"].str.split("[").str[0].str.strip()
    long["value"] = to_numeric(long["value"])
    long["series"] = long.get("series", pd.Series(dtype="object")).astype("string").str.strip(" ,")
    long = long.dropna(subset=["value", "date"])

    if frequency:
        long = long[long["frequency"] == frequency.upper()]

    sort_columns = [c for c in ("country_code", "series_code", "date") if c in long.columns]
    return long.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def economic_series(
    series_code: str,
    country_code: str | None = None,
    frequency: str = "M",
    path: Path | str | None = None,
    settings: Settings | None = None,
) -> pd.Series:
    """Extract a single indicator as a date-indexed :class:`pandas.Series`."""
    long = load_economic_indicators(path=path, frequency=frequency, settings=settings)
    mask = long["series_code"].astype("string").str.strip() == series_code
    if country_code and "country_code" in long.columns:
        mask &= long["country_code"].astype("string").str.strip() == country_code
    selected = long[mask]
    if selected.empty:
        raise ValueError(
            f"No observations for series {series_code!r}"
            + (f" and country {country_code!r}" if country_code else "")
        )
    out = selected.set_index("date")["value"].sort_index()
    out.name = series_code
    return out


# --------------------------------------------------------------------------
# RBI "Select Economic Indicators" workbook
# --------------------------------------------------------------------------


def _is_blank_row(row: pd.Series) -> bool:
    return bool(row.isna().all())


_NUMBERING_RE = re.compile(r"^(\d+(?:\.\d+)*)")


def _label_number(text: str) -> str | None:
    match = _NUMBERING_RE.match(text)
    return match.group(1) if match else None


def _is_child_label(parent: str, candidate: str) -> bool:
    """RBI labels are numbered (``2`` -> ``2.1`` -> ``2.1.1``).

    A forward-filled cell that breaks the numbering belongs to another section
    and must not be glued onto this column's name.
    """
    parent_number = _label_number(parent)
    candidate_number = _label_number(candidate)
    if parent_number is None or candidate_number is None:
        return True
    return candidate_number.startswith(f"{parent_number}.")


def _flatten_headers(header_block: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (column names, full hierarchical paths) for a merged header block.

    Every header row except the deepest one is forward-filled to undo Excel's
    merged cells; the deepest row is left alone because its cells are already
    per-column.  The column name is the most specific label, which keeps names
    short while ``paths`` preserves the whole section hierarchy.
    """
    rows = [header_block.iloc[i] for i in range(len(header_block))]
    filled = [row.ffill() for row in rows[:-1]] + rows[-1:] if rows else []

    names: list[str] = []
    paths: list[str] = []
    counts: dict[str, int] = {}
    for column in header_block.columns:
        parts: list[str] = []
        for original_row, row in zip(rows, filled):
            value = row[column]
            text = "" if pd.isna(value) else str(value).strip()
            if not text or text in parts:
                continue
            if parts and not _is_child_label(parts[-1], text):
                if pd.isna(original_row[column]):
                    # A forward-filled label from a neighbouring section.
                    continue
                # The cell has its own value, so it starts a new hierarchy.
                parts = []
            parts.append(text)
        paths.append(" / ".join(parts))
        if parts:
            deepest = re.sub(r"^\d+(?:\.\d+)*[a-zA-Z]?\b", "", parts[-1]).strip(" .:-")
            name = snake_case(deepest) or snake_case(parts[-1])
        else:
            name = f"column_{column}"
        counts[name] = counts.get(name, 0) + 1
        names.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return names, paths


def _label_position(raw: pd.DataFrame) -> tuple[int, int]:
    """Find (row, column) of the ``Year``/``Date`` header cell."""
    for row_index in range(min(len(raw), 30)):
        for column_index, value in enumerate(raw.iloc[row_index]):
            if isinstance(value, str) and value.strip().lower() in {"year", "date"}:
                return row_index, column_index
    raise ValueError("Could not locate a 'Year' or 'Date' header cell in the sheet")


def _first_data_row(raw: pd.DataFrame, label_row: int, label_column: int) -> int:
    for row_index in range(label_row + 1, len(raw)):
        value = raw.iat[row_index, label_column]
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return row_index
        text = "" if pd.isna(value) else str(value).strip()
        if re.match(r"^\d{4}(-\d{2,4})?$", text):
            return row_index
    raise ValueError("Could not locate the first data row in the sheet")


def load_rbi_indicators(
    path: Path | str | None = None,
    sheet: str = "Quarterly",
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Parse one sheet of the RBI indicators workbook into a tidy wide frame.

    The workbook uses a multi-row merged header and sparse year labels; both
    sheet layouts (``Quarterly`` and ``Monthly``) are detected automatically.
    The result has a ``period`` column plus one float column per indicator.
    """
    settings = settings or get_settings()
    xlsx_path = _resolve(path, settings.rbi_path)
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)

    label_row, label_column = _label_position(raw)
    data_row = _first_data_row(raw, label_row, label_column)

    header_start = data_row
    while header_start > 0 and not _is_blank_row(raw.iloc[header_start - 1]):
        header_start -= 1

    header_block = raw.iloc[header_start:data_row]
    names, paths = _flatten_headers(header_block)
    frame = raw.iloc[data_row:].copy()
    frame.columns = names

    label_name = names[label_column]
    label = frame[label_name]
    with warnings.catch_warnings():
        # Mixed year labels ('2024-25') and timestamps live in the same column.
        warnings.simplefilter("ignore", UserWarning)
        as_datetime = pd.to_datetime(label, errors="coerce")
    drop = [label_name] + [c for c in frame.columns if c.startswith("column_")]

    if as_datetime.notna().sum() >= max(3, 0.5 * label.notna().sum()):
        # Monthly layout: one full date per row.
        keep = as_datetime.notna()
        frame = frame[keep].copy()
        dates = as_datetime[keep]
        frame = frame.drop(columns=[c for c in dict.fromkeys(drop) if c in frame.columns])
        frame.insert(0, "date", dates.to_numpy())
        frame.insert(0, "period", dates.dt.strftime("%Y-%m-%d").to_numpy())
    else:
        # Quarterly layout: a sparse (merged) year column plus a quarter column.
        sub_name = names[label_column + 1] if label_column + 1 < len(names) else None
        years = label.ffill()
        if sub_name is None:
            periods = years.astype("string").str.strip()
        else:
            drop.append(sub_name)
            keep = frame[sub_name].notna()
            frame = frame[keep].copy()
            periods = (
                years.loc[frame.index].astype("string").str.strip()
                + " "
                + frame.loc[keep, sub_name].astype("string").str.strip()
            )
        frame = frame.drop(columns=[c for c in dict.fromkeys(drop) if c in frame.columns])
        frame.insert(0, "period", periods.to_numpy())
        frame = frame[frame["period"].notna()]

    for column in [c for c in frame.columns if c not in {"period", "date"}]:
        frame[column] = to_numeric(frame[column])
    frame = frame.dropna(axis=1, how="all")

    value_columns = [c for c in frame.columns if c not in {"period", "date"}]
    if not value_columns:
        raise ValueError(f"Sheet {sheet!r} contains no numeric indicator columns")
    frame = frame.dropna(subset=value_columns, how="all")

    lead = [c for c in ("period", "date") if c in frame.columns]
    ordered = lead + [c for c in frame.columns if c not in lead]
    frame = frame[ordered].reset_index(drop=True)
    frame.attrs["indicator_paths"] = {
        name: path for name, path in zip(names, paths) if name in frame.columns
    }
    return frame


# --------------------------------------------------------------------------
# Synthetic data (demo runs and tests never touch the real datasets)
# --------------------------------------------------------------------------


def synthetic_price_series(
    periods: int = 1000,
    seed: int = 42,
    start_price: float = 100.0,
    drift: float = 0.05,
    seasonal_amplitude: float = 10.0,
    seasonal_cycles: float = 10.0,
    noise: float = 2.0,
    start_date: str = "2018-01-01",
) -> pd.Series:
    """Deterministic trend + seasonality + noise series on business days."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start_date, periods=periods, name="date")
    trend = np.linspace(start_price, start_price * (1.0 + drift * periods / 252.0), periods)
    seasonality = seasonal_amplitude * np.sin(np.linspace(0.0, seasonal_cycles * 2 * np.pi, periods))
    shocks = rng.normal(0.0, noise, periods)
    values = trend + seasonality + shocks
    return pd.Series(values, index=index, name="close")


def synthetic_ohlcv(
    periods: int = 1000,
    seed: int = 42,
    ticker: str = "SYNTH",
    **kwargs,
) -> pd.DataFrame:
    """Build a full OHLCV frame around :func:`synthetic_price_series`."""
    close = synthetic_price_series(periods=periods, seed=seed, **kwargs)
    rng = np.random.default_rng(seed + 1)
    spread = np.abs(rng.normal(0.0, 0.6, periods)) + 0.1
    frame = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]).to_numpy(),
            "high": close.to_numpy() + spread,
            "low": close.to_numpy() - spread,
            "close": close.to_numpy(),
            "volume": rng.integers(1_000_000, 5_000_000, periods).astype("float64"),
            "ticker": ticker,
        },
        index=close.index,
    )
    frame["high"] = frame[["high", "open", "close"]].max(axis=1)
    frame["low"] = frame[["low", "open", "close"]].min(axis=1)
    return frame
