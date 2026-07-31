"""Turn a time-indexed feature table into supervised LSTM sequences.

The split is strictly chronological and the scalers are fitted on training rows
only.  Windows that would straddle a split boundary are dropped, so no test
observation can leak into a training sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(n, lookback, n_features)`` windows and ``(n, horizon)`` targets.

    Window ``i`` covers rows ``[i, i + lookback)`` and predicts rows
    ``[i + lookback, i + lookback + horizon)``.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    features = np.asarray(features, dtype="float64")
    targets = np.asarray(targets, dtype="float64").reshape(-1)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same number of rows")

    usable = len(features) - lookback - horizon + 1
    if usable <= 0:
        raise ValueError(
            f"Need at least {lookback + horizon} rows to build a window, got {len(features)}"
        )

    x = np.stack([features[i : i + lookback] for i in range(usable)])
    y = np.stack([targets[i + lookback : i + lookback + horizon] for i in range(usable)])
    return x, y


def chronological_split(
    n_rows: int,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> tuple[slice, slice, slice]:
    """Split ``n_rows`` into ordered train/validation/test row ranges."""
    if not 0.0 <= test_fraction < 1.0 or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("fractions must be in [0, 1)")
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("test_fraction + validation_fraction must be < 1")

    n_test = int(round(n_rows * test_fraction))
    n_validation = int(round(n_rows * validation_fraction))
    n_train = n_rows - n_test - n_validation
    if n_train <= 0:
        raise ValueError("Not enough rows for a training split")

    train_end = n_train
    validation_end = n_train + n_validation
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, n_rows)


@dataclass
class SequenceDataset:
    """Scaled sequences for every split, plus the fitted scalers."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_names: list[str]
    target_name: str
    lookback: int
    horizon: int
    test_index: pd.Index = field(default_factory=pd.Index)

    @property
    def n_features(self) -> int:
        return self.x_train.shape[2]

    @property
    def has_validation(self) -> bool:
        return len(self.x_validation) > 0

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Map scaled target values back to the original price scale."""
        values = np.asarray(values, dtype="float64")
        flat = values.reshape(-1, 1)
        restored = self.target_scaler.inverse_transform(flat)
        return restored.reshape(values.shape)

    def last_window(self) -> np.ndarray:
        """The most recent scaled window, ready for recursive forecasting."""
        source = self.x_test if len(self.x_test) else self.x_train
        return source[-1].copy()

    def previous_actual(self, split: str = "test") -> np.ndarray:
        """Last observed target of each window, in original units.

        Useful as a persistence baseline and for directional accuracy.
        """
        arrays = {"train": self.x_train, "validation": self.x_validation, "test": self.x_test}
        if split not in arrays:
            raise KeyError(f"unknown split {split!r}")
        x = arrays[split]
        if len(x) == 0:
            return np.empty((0,))
        target_position = self.feature_names.index(self.target_name)
        scaled_last = x[:, -1, target_position]
        column = np.zeros((len(scaled_last), len(self.feature_names)))
        column[:, target_position] = scaled_last
        return self.feature_scaler.inverse_transform(column)[:, target_position]


def build_dataset(
    frame: pd.DataFrame,
    target_column: str = "close",
    lookback: int = 60,
    horizon: int = 1,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> SequenceDataset:
    """Scale, window and split a feature frame into a :class:`SequenceDataset`."""
    if target_column not in frame.columns:
        raise KeyError(f"{target_column!r} is not a column of the frame")

    numeric = frame.select_dtypes(include=[np.number]).astype("float64")
    if target_column not in numeric.columns:
        raise TypeError(f"target column {target_column!r} must be numeric")
    if numeric.isna().to_numpy().any():
        raise ValueError("frame still contains NaNs; drop or impute them before windowing")

    values = numeric.to_numpy()
    train_rows, validation_rows, test_rows = chronological_split(
        len(numeric), test_fraction, validation_fraction
    )

    feature_scaler = MinMaxScaler(feature_range=(0, 1)).fit(values[train_rows])
    scaled = feature_scaler.transform(values)

    target_position = list(numeric.columns).index(target_column)
    target_scaler = MinMaxScaler(feature_range=(0, 1)).fit(
        values[train_rows, target_position].reshape(-1, 1)
    )
    scaled_target = target_scaler.transform(values[:, target_position].reshape(-1, 1)).reshape(-1)

    def window_split(rows: slice) -> tuple[np.ndarray, np.ndarray]:
        block = scaled[rows]
        target_block = scaled_target[rows]
        if len(block) < lookback + horizon:
            empty_x = np.empty((0, lookback, scaled.shape[1]))
            return empty_x, np.empty((0, horizon))
        return make_sequences(block, target_block, lookback, horizon)

    x_train, y_train = window_split(train_rows)
    x_validation, y_validation = window_split(validation_rows)
    x_test, y_test = window_split(test_rows)

    test_labels = frame.index[test_rows]
    test_index = test_labels[lookback + horizon - 1 :][: len(x_test)]

    return SequenceDataset(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=list(numeric.columns),
        target_name=target_column,
        lookback=lookback,
        horizon=horizon,
        test_index=pd.Index(test_index),
    )
