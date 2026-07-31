"""The LSTM forecaster.

TensorFlow is imported lazily so that the data, feature and metric modules stay
usable (and testable) in environments where it is not installed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def tensorflow_available() -> bool:
    """True when TensorFlow can be imported in this interpreter."""
    try:  # pragma: no cover - depends on the environment
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


def _keras():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required to build or train the LSTM. "
            "Install it with: pip install 'tensorflow>=2.15,<2.20'"
        ) from error
    return keras


def set_seed(seed: int) -> None:
    """Seed python, numpy and TensorFlow for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tensorflow_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


@dataclass
class LSTMConfig:
    """Architecture and optimisation hyper-parameters."""

    lookback: int = 60
    horizon: int = 1
    n_features: int = 1
    units: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    dense_units: int = 16
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 32
    patience: int = 6
    seed: int = 42

    def __post_init__(self) -> None:
        if self.lookback < 1 or self.horizon < 1 or self.n_features < 1:
            raise ValueError("lookback, horizon and n_features must be >= 1")
        if not self.units:
            raise ValueError("at least one LSTM layer is required")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class FinancialLSTM:
    """Stacked LSTM regressor for financial time series."""

    config: LSTMConfig = field(default_factory=LSTMConfig)
    model: Any = None

    @classmethod
    def from_dataset(cls, dataset, **overrides) -> "FinancialLSTM":
        """Build a model sized for a :class:`~lstm_financial.windows.SequenceDataset`."""
        config = LSTMConfig(
            lookback=dataset.lookback,
            horizon=dataset.horizon,
            n_features=dataset.n_features,
            **overrides,
        )
        return cls(config=config)

    def build(self) -> Any:
        """Create and compile the Keras model."""
        keras = _keras()
        set_seed(self.config.seed)

        layers: list[Any] = [keras.layers.Input(shape=(self.config.lookback, self.config.n_features))]
        for position, units in enumerate(self.config.units):
            last = position == len(self.config.units) - 1
            layers.append(keras.layers.LSTM(units, return_sequences=not last))
            if self.config.dropout:
                layers.append(keras.layers.Dropout(self.config.dropout))
        if self.config.dense_units:
            layers.append(keras.layers.Dense(self.config.dense_units, activation="relu"))
        layers.append(keras.layers.Dense(self.config.horizon))

        model = keras.Sequential(layers, name="financial_lstm")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )
        self.model = model
        return model

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """Train the model, returning the Keras history dictionary."""
        keras = _keras()
        if self.model is None:
            self.build()

        callbacks = []
        monitor = "val_loss" if validation_data is not None and len(validation_data[0]) else "loss"
        if self.config.patience:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=self.config.patience,
                    restore_best_weights=True,
                )
            )
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    factor=0.5,
                    patience=max(1, self.config.patience // 2),
                    min_lr=1e-5,
                )
            )

        if validation_data is not None and not len(validation_data[0]):
            validation_data = None

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs or self.config.epochs,
            batch_size=batch_size or self.config.batch_size,
            validation_data=validation_data,
            shuffle=False,
            callbacks=callbacks,
            verbose=verbose,
        )
        return {name: [float(v) for v in values] for name, values in history.history.items()}

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Scaled predictions of shape ``(n, horizon)``."""
        if self.model is None:
            raise RuntimeError("Model is not built or trained yet; call fit() first")
        return np.asarray(self.model.predict(x, verbose=verbose), dtype="float64")

    def forecast(self, last_window: np.ndarray, steps: int = 30) -> np.ndarray:
        """Recursively forecast ``steps`` scaled values beyond the last window.

        Only defined for univariate models: with extra features there is no way
        to synthesise the exogenous inputs of the next step.
        """
        if self.model is None:
            raise RuntimeError("Model is not built or trained yet; call fit() first")
        if steps < 1:
            raise ValueError("steps must be >= 1")

        window = np.asarray(last_window, dtype="float64").reshape(self.config.lookback, -1)
        if window.shape[1] != 1:
            raise ValueError(
                "Recursive forecasting requires a univariate model; "
                f"this one takes {window.shape[1]} features. Train with a single "
                "feature column or predict one step at a time from real inputs."
            )

        predictions: list[float] = []
        while len(predictions) < steps:
            batch = window.reshape(1, self.config.lookback, 1)
            step = self.predict(batch)[0]
            for value in np.atleast_1d(step):
                if len(predictions) >= steps:
                    break
                predictions.append(float(value))
                window = np.vstack([window[1:], [[float(value)]]])
        return np.asarray(predictions, dtype="float64")

    def save(self, path: Path | str) -> Path:
        """Persist the trained model in the native Keras format."""
        if self.model is None:
            raise RuntimeError("Nothing to save; build or train the model first")
        target = Path(path)
        if target.suffix != ".keras":
            target = target.with_suffix(".keras")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(target)
        return target

    @classmethod
    def load(cls, path: Path | str, config: LSTMConfig | None = None) -> "FinancialLSTM":
        """Load a model saved by :meth:`save`."""
        keras = _keras()
        model = keras.models.load_model(Path(path))
        if config is None:
            _, lookback, n_features = model.input_shape
            horizon = model.output_shape[-1]
            config = LSTMConfig(lookback=lookback, horizon=horizon, n_features=n_features)
        return cls(config=config, model=model)

    def summary_lines(self) -> Sequence[str]:
        """Model summary as a list of strings (handy for logs)."""
        if self.model is None:
            raise RuntimeError("Model is not built yet")
        lines: list[str] = []
        self.model.summary(print_fn=lines.append)
        return lines


def persistence_forecast(previous: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Naive baseline: tomorrow equals today, repeated over the horizon."""
    previous = np.asarray(previous, dtype="float64").reshape(-1, 1)
    return np.repeat(previous, horizon, axis=1)
