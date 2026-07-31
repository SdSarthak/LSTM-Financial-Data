"""End-to-end training pipeline: features -> windows -> LSTM -> report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURE_COLUMNS, add_technical_indicators, select_features
from .metrics import format_metrics, regression_metrics
from .model import FinancialLSTM, LSTMConfig, persistence_forecast
from .plots import plot_forecast, plot_predictions, plot_training_history
from .windows import SequenceDataset, build_dataset


@dataclass
class ForecastRun:
    """Everything a single training run produced."""

    name: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    history: dict[str, list[float]]
    predictions: np.ndarray
    actuals: np.ndarray
    index: pd.Index
    future_forecast: np.ndarray | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)

    def report(self) -> str:
        """Human readable summary of the run."""
        lines = [
            f"Run: {self.name}",
            f"Test windows: {len(self.actuals)}",
            "",
            "LSTM:",
            format_metrics(self.metrics),
            "",
            "Persistence baseline:",
            format_metrics(self.baseline_metrics),
        ]
        if self.future_forecast is not None and len(self.future_forecast):
            forecast = self.future_forecast
            lines += [
                "",
                f"Recursive forecast ({len(forecast)} steps): "
                f"first {forecast[0]:.2f}, last {forecast[-1]:.2f}, "
                f"min {forecast.min():.2f}, max {forecast.max():.2f}",
            ]
        if self.artifacts:
            lines += ["", "Artifacts:"] + [f"  {k}: {v}" for k, v in self.artifacts.items()]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "config": self.config,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "epochs_run": len(next(iter(self.history.values()), [])),
            "test_windows": int(len(self.actuals)),
            "artifacts": self.artifacts,
        }


def prepare_features(
    frame: pd.DataFrame,
    price_column: str = "close",
    feature_columns: Sequence[str] | None = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Add technical indicators and keep the modelling columns."""
    enriched = add_technical_indicators(frame, price_column=price_column)
    if feature_columns is None:
        return enriched.select_dtypes(include=[np.number])
    return select_features(enriched, list(feature_columns))


def run_forecast(
    frame: pd.DataFrame,
    name: str = "run",
    target_column: str = "close",
    feature_columns: Sequence[str] | None = DEFAULT_FEATURE_COLUMNS,
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
    """Train an LSTM on ``frame`` and evaluate it against a persistence baseline.

    ``frame`` must be a date-indexed OHLCV-style table containing
    ``target_column``.  Set ``add_indicators=False`` to model the raw columns.
    """
    modelling_frame = (
        prepare_features(frame, price_column=target_column, feature_columns=feature_columns)
        if add_indicators
        else frame.select_dtypes(include=[np.number]).dropna()
    )

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
    validation = (
        (dataset.x_validation, dataset.y_validation) if dataset.has_validation else None
    )
    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=validation,
        verbose=verbose,
    )

    scaled_predictions = model.predict(dataset.x_test)
    predictions = dataset.inverse_target(scaled_predictions)
    actuals = dataset.inverse_target(dataset.y_test)
    previous = dataset.previous_actual("test")

    first_step_prediction = predictions[:, 0]
    first_step_actual = actuals[:, 0]
    metrics = regression_metrics(first_step_actual, first_step_prediction, previous=previous)
    baseline = persistence_forecast(previous, horizon=dataset.horizon)[:, 0]
    baseline_metrics = regression_metrics(first_step_actual, baseline, previous=previous)

    future: np.ndarray | None = None
    if forecast_steps and dataset.n_features == 1:
        scaled_future = model.forecast(dataset.last_window(), steps=forecast_steps)
        future = dataset.inverse_target(scaled_future).reshape(-1)

    artifacts: dict[str, str] = {}
    if artifacts_dir is not None:
        directory = Path(artifacts_dir)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts["predictions_plot"] = str(
            plot_predictions(
                first_step_actual,
                first_step_prediction,
                directory / f"{name}_predictions.png",
                index=dataset.test_index if len(dataset.test_index) == len(actuals) else None,
                title=f"{name}: LSTM predictions vs actual",
            )
        )
        artifacts["history_plot"] = str(
            plot_training_history(history, directory / f"{name}_history.png", title=f"{name}: loss")
        )
        if future is not None:
            artifacts["forecast_plot"] = str(
                plot_forecast(
                    first_step_actual[-min(len(first_step_actual), 120) :],
                    future,
                    directory / f"{name}_forecast.png",
                    title=f"{name}: {len(future)}-step recursive forecast",
                )
            )
        artifacts["model"] = str(model.save(directory / f"{name}_model.keras"))

    run = ForecastRun(
        name=name,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        history=history,
        predictions=predictions,
        actuals=actuals,
        index=dataset.test_index,
        future_forecast=future,
        artifacts=artifacts,
        config={
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
