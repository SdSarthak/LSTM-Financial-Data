"""End-to-end training pipeline: features -> windows -> LSTM -> report.

Two modelling targets are supported:

``return`` (default)
    The network predicts the next log return and the price path is
    reconstructed from the last observed price.  Returns are roughly
    stationary, so a scaler fitted on the training block still fits the test
    block years later.

``level``
    The network predicts the price itself.  Simpler to read, but on a trending
    series the test prices fall outside the range the scaler was fitted on and
    the model cannot extrapolate - expect it to lose to the naive baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .features import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_RETURN_FEATURE_COLUMNS,
    add_technical_indicators,
    log_returns,
    select_features,
)
from .metrics import format_metrics, regression_metrics
from .model import FinancialLSTM, LSTMConfig, persistence_forecast
from .plots import plot_forecast, plot_predictions, plot_training_history
from .windows import SequenceDataset, build_dataset

TARGET_MODES = ("return", "level")


@dataclass
class ForecastRun:
    """Everything a single training run produced (prices, not scaled units)."""

    name: str
    target_mode: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    history: dict[str, list[float]]
    predictions: np.ndarray
    actuals: np.ndarray
    index: pd.Index
    future_forecast: np.ndarray | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)

    def beats_baseline(self) -> bool:
        """True when the LSTM's RMSE is lower than the persistence baseline's."""
        return self.metrics["RMSE"] < self.baseline_metrics["RMSE"]

    def report(self) -> str:
        """Human readable summary of the run."""
        lines = [
            f"Run: {self.name} (target mode: {self.target_mode})",
            f"Test windows: {len(self.actuals)}",
            "",
            "LSTM:",
            format_metrics(self.metrics),
            "",
            "Persistence baseline (tomorrow = today):",
            format_metrics(self.baseline_metrics),
            "",
            "LSTM beats the baseline on RMSE: " + ("yes" if self.beats_baseline() else "no"),
        ]
        if self.future_forecast is not None and len(self.future_forecast):
            forecast = self.future_forecast
            lines += [
                "",
                f"Recursive forecast ({len(forecast)} steps): "
                f"first {forecast[0]:.2f}, last {forecast[-1]:.2f}, "
                f"min {forecast.min():.2f}, max {forecast.max():.2f}",
            ]
        elif len(self.config.get("features", [])) > 1:
            lines += [
                "",
                "Recursive forecast: skipped - it needs a univariate model. "
                "Re-run with a single --features column, or use --horizon for "
                "direct multi-step forecasting.",
            ]
        if self.artifacts:
            lines += ["", "Artifacts:"] + [f"  {k}: {v}" for k, v in self.artifacts.items()]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target_mode": self.target_mode,
            "config": self.config,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "beats_baseline": self.beats_baseline(),
            "epochs_run": len(next(iter(self.history.values()), [])),
            "test_windows": int(len(self.actuals)),
            "artifacts": self.artifacts,
        }


def enrich(
    frame: pd.DataFrame,
    price_column: str = "close",
    add_indicators: bool = True,
) -> pd.DataFrame:
    """Indicator columns (optional) plus a guaranteed ``log_return`` column."""
    if add_indicators:
        return add_technical_indicators(frame, price_column=price_column)

    enriched = frame.copy()
    if "log_return" not in enriched.columns:
        enriched["log_return"] = log_returns(enriched[price_column].astype("float64"))
    return enriched.dropna(subset=[price_column, "log_return"])


def prepare_features(
    frame: pd.DataFrame,
    price_column: str = "close",
    feature_columns: Sequence[str] | None = DEFAULT_FEATURE_COLUMNS,
    add_indicators: bool = True,
) -> pd.DataFrame:
    """Add technical indicators (optional) and keep the modelling columns."""
    enriched = enrich(frame, price_column=price_column, add_indicators=add_indicators)
    if feature_columns is None:
        # Columns the source leaves empty (e.g. "capital_gains") carry no signal
        # and would otherwise block windowing, which rejects NaNs.
        return enriched.select_dtypes(include=[np.number]).dropna(axis=1, how="all").dropna()
    return select_features(enriched, list(feature_columns))


def _default_features(target_mode: str) -> tuple[str, ...]:
    return DEFAULT_FEATURE_COLUMNS if target_mode == "level" else DEFAULT_RETURN_FEATURE_COLUMNS


def _previous_prices(prices: pd.Series, index: pd.Index, horizon: int) -> np.ndarray:
    """Price observed just before each test window's first target step."""
    positions = prices.index.get_indexer(index)
    if (positions < horizon).any():
        raise ValueError("cannot align the test windows with their preceding prices")
    return prices.to_numpy()[positions - horizon]


def run_forecast(
    frame: pd.DataFrame,
    name: str = "run",
    price_column: str = "close",
    target_mode: str = "return",
    feature_columns: Sequence[str] | None = None,
    lookback: int = 60,
    horizon: int = 1,
    epochs: int = 30,
    batch_size: int = 32,
    units: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    patience: int = 6,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    forecast_steps: int = 30,
    artifacts_dir: Path | str | None = None,
    verbose: int = 1,
    add_indicators: bool = True,
) -> ForecastRun:
    """Train an LSTM on ``frame`` and score it against a persistence baseline.

    ``frame`` must be a date-indexed table containing ``price_column``.  All
    reported metrics are in price units, so the two target modes and the naive
    baseline are directly comparable.
    """
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    if feature_columns is None:
        feature_columns = _default_features(target_mode)

    target_column = price_column if target_mode == "level" else "log_return"
    columns = tuple(dict.fromkeys((*feature_columns, target_column)))
    enriched = enrich(frame, price_column=price_column, add_indicators=add_indicators)
    modelling_frame = select_features(enriched, list(columns))
    prices = enriched[price_column].astype("float64")

    dataset: SequenceDataset = build_dataset(
        modelling_frame,
        target_column=target_column,
        lookback=lookback,
        horizon=horizon,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )
    if len(dataset.x_test) == 0:
        raise ValueError(
            "The test split is too short for the chosen lookback/horizon; "
            "use more history, a smaller lookback or a larger test_fraction."
        )

    model = FinancialLSTM(
        config=LSTMConfig(
            lookback=dataset.lookback,
            horizon=dataset.horizon,
            n_features=dataset.n_features,
            units=tuple(units),
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
        )
    )
    validation = (dataset.x_validation, dataset.y_validation) if dataset.has_validation else None
    history = model.fit(dataset.x_train, dataset.y_train, validation_data=validation, verbose=verbose)

    predicted_target = dataset.inverse_target(model.predict(dataset.x_test))
    actual_target = dataset.inverse_target(dataset.y_test)
    previous = _previous_prices(prices, dataset.test_index, dataset.horizon)

    if target_mode == "return":
        # Rebuild the price path from the last observed price and the returns.
        base = previous.reshape(-1, 1)
        predictions = base * np.exp(np.cumsum(predicted_target, axis=1))
        actuals = base * np.exp(np.cumsum(actual_target, axis=1))
    else:
        predictions = predicted_target
        actuals = actual_target

    metrics = regression_metrics(actuals[:, 0], predictions[:, 0], previous=previous)
    baseline = persistence_forecast(previous, horizon=dataset.horizon)[:, 0]
    baseline_metrics = regression_metrics(actuals[:, 0], baseline, previous=previous)

    future: np.ndarray | None = None
    if forecast_steps and dataset.n_features == 1:
        future_target = dataset.inverse_target(
            model.forecast(dataset.last_window(), steps=forecast_steps)
        ).reshape(-1)
        future = (
            float(prices.iloc[-1]) * np.exp(np.cumsum(future_target))
            if target_mode == "return"
            else future_target
        )

    artifacts: dict[str, str] = {}
    if artifacts_dir is not None:
        directory = Path(artifacts_dir)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts["predictions_plot"] = str(
            plot_predictions(
                actuals[:, 0],
                predictions[:, 0],
                directory / f"{name}_predictions.png",
                index=dataset.test_index if len(dataset.test_index) == len(actuals) else None,
                title=f"{name}: LSTM predictions vs actual ({target_mode} target)",
            )
        )
        artifacts["history_plot"] = str(
            plot_training_history(history, directory / f"{name}_history.png", title=f"{name}: loss")
        )
        if future is not None:
            artifacts["forecast_plot"] = str(
                plot_forecast(
                    actuals[-min(len(actuals), 120) :, 0],
                    future,
                    directory / f"{name}_forecast.png",
                    title=f"{name}: {len(future)}-step recursive forecast",
                )
            )
        artifacts["model"] = str(model.save(directory / f"{name}_model.keras"))

    run = ForecastRun(
        name=name,
        target_mode=target_mode,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        history=history,
        predictions=predictions,
        actuals=actuals,
        index=dataset.test_index,
        future_forecast=future,
        artifacts=artifacts,
        config={
            "price_column": price_column,
            "target_column": target_column,
            "features": dataset.feature_names,
            "lookback": lookback,
            "horizon": horizon,
            "epochs": epochs,
            "batch_size": batch_size,
            "units": list(units),
            "dropout": dropout,
            "learning_rate": learning_rate,
            "seed": seed,
            "train_windows": int(len(dataset.x_train)),
            "validation_windows": int(len(dataset.x_validation)),
        },
    )

    if artifacts_dir is not None:
        summary_path = Path(artifacts_dir) / f"{name}_summary.json"
        summary_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        run.artifacts["summary"] = str(summary_path)
    return run
