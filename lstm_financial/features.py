"""Technical indicators used as LSTM input features.

Everything here is a pure pandas transformation of a price series, which keeps
the feature set reproducible and unit-testable without any market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature set for modelling the price level directly.
DEFAULT_FEATURE_COLUMNS = (
    "close",
    "log_return",
    "sma_10",
    "sma_30",
    "rsi_14",
    "macd",
    "macd_histogram",
    "volatility_20",
    "bb_percent_b",
)

# Feature set for modelling log returns.  It deliberately excludes price levels
# and moving averages: those trend out of the range the scaler saw during
# training, and the network cannot extrapolate beyond it.
DEFAULT_RETURN_FEATURE_COLUMNS = (
    "log_return",
    "rsi_14",
    "macd_histogram",
    "volatility_20",
    "bb_percent_b",
)


def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Rolling arithmetic mean over ``window`` observations."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return prices.rolling(window=window, min_periods=window).mean().rename(f"sma_{window}")


def exponential_moving_average(prices: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted mean with the conventional ``span`` decay."""
    if span < 1:
        raise ValueError("span must be >= 1")
    return prices.ewm(span=span, adjust=False).mean().rename(f"ema_{span}")


def log_returns(prices: pd.Series) -> pd.Series:
    """Log return of consecutive observations; first value is NaN."""
    positive = prices.where(prices > 0)
    return np.log(positive).diff().rename("log_return")


def relative_strength_index(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in the 0-100 range."""
    if period < 1:
        raise ValueError("period must be >= 1")
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rsi = pd.Series(np.nan, index=prices.index, dtype="float64")
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    no_loss = (avg_loss == 0) & (avg_gain > 0)
    normal = avg_loss > 0
    rs = avg_gain[normal] / avg_loss[normal]
    rsi[normal] = 100.0 - (100.0 / (1.0 + rs))
    rsi[no_loss] = 100.0
    rsi[both_zero] = 50.0
    return rsi.rename(f"rsi_{period}")


def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence line, signal and histogram."""
    if fast >= slow:
        raise ValueError("fast span must be shorter than slow span")
    macd_line = exponential_moving_average(prices, fast) - exponential_moving_average(prices, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_line - signal_line,
        }
    )


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Middle/upper/lower bands plus the normalised %B position."""
    middle = prices.rolling(window=window, min_periods=window).mean()
    deviation = prices.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * deviation
    lower = middle - num_std * deviation
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_percent_b": (prices - lower) / width,
        }
    )


def realised_volatility(prices: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualised rolling standard deviation of log returns."""
    returns = log_returns(prices)
    rolling = returns.rolling(window=window, min_periods=window).std(ddof=0)
    return (rolling * np.sqrt(periods_per_year)).rename(f"volatility_{window}")


def add_technical_indicators(
    frame: pd.DataFrame,
    price_column: str = "close",
    dropna: bool = True,
) -> pd.DataFrame:
    """Append the standard indicator set to an OHLCV frame.

    Rows where any indicator is still warming up are dropped by default so the
    result can go straight into a model without NaN handling downstream.
    """
    if price_column not in frame.columns:
        raise KeyError(f"{price_column!r} is not a column of the frame: {list(frame.columns)}")

    prices = frame[price_column].astype("float64")
    original_columns = list(frame.columns)
    enriched = frame.copy()
    enriched["log_return"] = log_returns(prices)
    enriched["sma_10"] = simple_moving_average(prices, 10)
    enriched["sma_30"] = simple_moving_average(prices, 30)
    enriched["ema_12"] = exponential_moving_average(prices, 12)
    enriched["ema_26"] = exponential_moving_average(prices, 26)
    enriched["rsi_14"] = relative_strength_index(prices, 14)
    enriched = enriched.join(macd(prices))
    enriched = enriched.join(bollinger_bands(prices))
    enriched["volatility_20"] = realised_volatility(prices)

    if dropna:
        # Only the warm-up of the indicators (and the price itself) may drop a
        # row: source columns such as an always-empty "capital_gains" must not.
        warmup_columns = [price_column] + [c for c in enriched.columns if c not in original_columns]
        enriched = enriched.dropna(subset=warmup_columns)
    return enriched


def select_features(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Keep the requested feature columns, failing loudly on unknown names."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return frame.loc[:, list(columns)]
