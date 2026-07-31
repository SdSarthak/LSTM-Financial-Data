"""Chart helpers.

Figures are always written to disk with a non-interactive backend so the
pipeline runs unattended; pass ``show=True`` for an interactive session.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib

if not os.environ.get("LSTM_FIN_INTERACTIVE_PLOTS"):
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _finish(figure: plt.Figure, output_path: Path | str, show: bool) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return path


def plot_series(
    series: pd.Series,
    output_path: Path | str,
    title: str = "Time series",
    ylabel: str = "Value",
    show: bool = False,
) -> Path:
    """Line chart of a single series."""
    figure, axes = plt.subplots(figsize=(13, 5))
    axes.plot(series.index, series.to_numpy(), color="#1f77b4", linewidth=1.2)
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.3)
    return _finish(figure, output_path, show)


def plot_predictions(
    actual: Sequence[float],
    predicted: Sequence[float],
    output_path: Path | str,
    index: Sequence | None = None,
    title: str = "Predictions vs actual",
    show: bool = False,
) -> Path:
    """Overlay predicted and observed values on the test window."""
    actual = np.asarray(actual, dtype="float64").reshape(-1)
    predicted = np.asarray(predicted, dtype="float64").reshape(-1)
    x = np.asarray(index) if index is not None and len(index) == len(actual) else np.arange(len(actual))

    figure, axes = plt.subplots(figsize=(13, 5))
    axes.plot(x, actual, label="Actual", color="#1f77b4", alpha=0.85, linewidth=1.3)
    axes.plot(x, predicted, label="Predicted", color="#d62728", alpha=0.85, linewidth=1.3)
    axes.set_title(title)
    axes.set_xlabel("Time")
    axes.set_ylabel("Price")
    axes.legend()
    axes.grid(True, alpha=0.3)
    return _finish(figure, output_path, show)


def plot_training_history(
    history: dict[str, Sequence[float]],
    output_path: Path | str,
    title: str = "Training history",
    show: bool = False,
) -> Path:
    """Loss curves from a Keras ``History.history`` dictionary."""
    figure, axes = plt.subplots(figsize=(9, 5))
    for name, values in history.items():
        if not name.endswith("loss"):
            continue
        axes.plot(range(1, len(values) + 1), values, label=name)
    axes.set_title(title)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.legend()
    axes.grid(True, alpha=0.3)
    return _finish(figure, output_path, show)


def plot_forecast(
    history: Sequence[float],
    forecast: Sequence[float],
    output_path: Path | str,
    title: str = "Recursive forecast",
    show: bool = False,
) -> Path:
    """Recent history followed by the multi-step forecast."""
    history = np.asarray(history, dtype="float64").reshape(-1)
    forecast = np.asarray(forecast, dtype="float64").reshape(-1)
    history_x = np.arange(len(history))
    forecast_x = np.arange(len(history), len(history) + len(forecast))

    figure, axes = plt.subplots(figsize=(13, 5))
    axes.plot(history_x, history, label="History", color="#1f77b4", linewidth=1.3)
    axes.plot(forecast_x, forecast, label="Forecast", color="#ff7f0e", linewidth=1.5)
    axes.axvline(len(history) - 1, color="grey", linestyle="--", alpha=0.6)
    axes.set_title(title)
    axes.set_xlabel("Time step")
    axes.set_ylabel("Price")
    axes.legend()
    axes.grid(True, alpha=0.3)
    return _finish(figure, output_path, show)
