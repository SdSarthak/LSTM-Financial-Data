"""LSTM forecasting toolkit for financial and macroeconomic time series."""

from .config import Settings, get_settings
from .features import add_technical_indicators, select_features
from .metrics import format_metrics, regression_metrics
from .model import FinancialLSTM, LSTMConfig, tensorflow_available
from .pipeline import ForecastRun, enrich, prepare_features, run_forecast
from .windows import SequenceDataset, build_dataset, chronological_split, make_sequences

__version__ = "0.2.0"

__all__ = [
    "ForecastRun",
    "FinancialLSTM",
    "LSTMConfig",
    "SequenceDataset",
    "Settings",
    "__version__",
    "add_technical_indicators",
    "build_dataset",
    "chronological_split",
    "enrich",
    "format_metrics",
    "get_settings",
    "make_sequences",
    "prepare_features",
    "regression_metrics",
    "run_forecast",
    "select_features",
    "tensorflow_available",
]
