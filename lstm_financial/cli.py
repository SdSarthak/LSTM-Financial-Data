"""Command line interface: ``python -m lstm_financial <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import datasets
from .config import Settings, get_settings
from .features import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_RETURN_FEATURE_COLUMNS,
    add_technical_indicators,
)
from .model import tensorflow_available
from .pipeline import TARGET_MODES, run_forecast
from .plots import plot_series

BANNER = "=" * 72


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lookback", type=int, default=60, help="window length in observations")
    parser.add_argument("--horizon", type=int, default=1, help="steps predicted per window")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units", type=int, nargs="+", default=[64, 32], help="LSTM layer sizes")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6, help="early stopping patience (0 disables)")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--forecast-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-mode",
        choices=list(TARGET_MODES),
        default="return",
        help="predict the next log return (default) or the price level directly",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "feature columns to feed the model; defaults to "
            f"{list(DEFAULT_RETURN_FEATURE_COLUMNS)} for --target-mode return and "
            f"{list(DEFAULT_FEATURE_COLUMNS)} for --target-mode level. "
            "A single feature enables the recursive multi-step forecast."
        ),
    )
    parser.add_argument("--artifacts-dir", default=None, help="where plots and the model are written")
    parser.add_argument("--quiet", action="store_true", help="silence Keras progress bars")


def _artifacts_dir(args: argparse.Namespace, settings: Settings) -> Path:
    directory = Path(args.artifacts_dir) if args.artifacts_dir else settings.artifacts_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _training_kwargs(args: argparse.Namespace) -> dict:
    return {
        "lookback": args.lookback,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "units": tuple(args.units),
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "forecast_steps": args.forecast_steps,
        "seed": args.seed,
        "target_mode": args.target_mode,
        "feature_columns": tuple(args.features) if args.features else None,
        "verbose": 0 if args.quiet else 1,
    }


def _require_tensorflow() -> None:
    if not tensorflow_available():
        raise SystemExit(
            "TensorFlow is not installed, so no model can be trained.\n"
            "Install it with: pip install 'tensorflow>=2.15,<2.20'\n"
            "The 'explore' and 'indicators' commands work without it."
        )


def command_demo(args: argparse.Namespace, settings: Settings) -> int:
    """Train on a deterministic synthetic series - no datasets required."""
    _require_tensorflow()
    frame = datasets.synthetic_ohlcv(periods=args.periods, seed=args.seed)
    print(BANNER)
    print(f"Synthetic demo: {len(frame)} business days, close "
          f"{frame['close'].min():.2f} - {frame['close'].max():.2f}")
    print(BANNER)
    run = run_forecast(
        frame,
        name="demo",
        artifacts_dir=_artifacts_dir(args, settings),
        **_training_kwargs(args),
    )
    print("\n" + run.report())
    return 0


def command_train(args: argparse.Namespace, settings: Settings) -> int:
    """Train on one ticker from the world stock prices dataset."""
    _require_tensorflow()
    history = datasets.load_ticker_history(args.ticker, path=args.stock_file, settings=settings)
    print(BANNER)
    print(f"{args.ticker}: {len(history)} rows, {history.index.min().date()} -> "
          f"{history.index.max().date()}")
    print(BANNER)
    run = run_forecast(
        history,
        name=args.ticker.lower(),
        artifacts_dir=_artifacts_dir(args, settings),
        **_training_kwargs(args),
    )
    print("\n" + run.report())
    return 0


def command_economic(args: argparse.Namespace, settings: Settings) -> int:
    """Train on a macroeconomic indicator series."""
    _require_tensorflow()
    series = datasets.economic_series(
        args.series_code,
        country_code=args.country,
        frequency=args.frequency,
        path=args.economic_file,
        settings=settings,
    )
    frame = series.to_frame(name="close")
    print(BANNER)
    print(f"{args.series_code} ({args.country or 'all countries'}): {len(frame)} observations, "
          f"{frame.index.min().date()} -> {frame.index.max().date()}")
    print(BANNER)
    kwargs = _training_kwargs(args)
    # Macro series are short and have no volume or intraday range, so the
    # technical indicator set does not apply: model the series itself.
    kwargs["feature_columns"] = ("close",) if args.target_mode == "level" else ("log_return",)
    run = run_forecast(
        frame,
        name=f"econ_{args.series_code.lower()}",
        add_indicators=False,
        artifacts_dir=_artifacts_dir(args, settings),
        **kwargs,
    )
    print("\n" + run.report())
    return 0


def command_explore(args: argparse.Namespace, settings: Settings) -> int:
    """Summarise whichever dataset is asked for, without training anything."""
    if args.dataset == "stocks":
        tickers = datasets.list_tickers(path=args.stock_file, settings=settings)
        print(f"{len(tickers)} tickers in {settings.stock_file}")
        print(", ".join(tickers[: args.limit]))
    elif args.dataset == "economic":
        long = datasets.load_economic_indicators(path=args.economic_file, settings=settings)
        print(f"{len(long)} observations, {long['series_code'].nunique()} series, "
              f"{long['country_code'].nunique()} countries")
        summary = (
            long.groupby(["series_code", "series"], dropna=False)["value"]
            .agg(["count", "min", "max"])
            .sort_values("count", ascending=False)
        )
        print(summary.head(args.limit).to_string())
    elif args.dataset == "rbi":
        frame = datasets.load_rbi_indicators(path=args.rbi_file, sheet=args.sheet, settings=settings)
        indicators = [c for c in frame.columns if c not in {"period", "date"}]
        print(f"sheet {args.sheet!r}: {len(frame)} periods, {len(indicators)} indicators")
        print(f"periods: {frame['period'].iloc[0]} -> {frame['period'].iloc[-1]}")
        paths = frame.attrs.get("indicator_paths", {})
        for column in indicators[: args.limit]:
            coverage = frame[column].notna().sum()
            print(f"  {column}  ({coverage} observations)  {paths.get(column, '')}")
    else:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"unknown dataset {args.dataset!r}")
    return 0


def command_indicators(args: argparse.Namespace, settings: Settings) -> int:
    """Write technical indicators for one ticker to a CSV (and optionally a plot)."""
    history = datasets.load_ticker_history(args.ticker, path=args.stock_file, settings=settings)
    enriched = add_technical_indicators(history)
    directory = _artifacts_dir(args, settings)
    output = Path(args.output) if args.output else directory / f"{args.ticker.lower()}_indicators.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output)
    print(f"wrote {len(enriched)} rows x {enriched.shape[1]} columns -> {output}")
    if args.plot:
        path = plot_series(
            enriched["close"],
            directory / f"{args.ticker.lower()}_close.png",
            title=f"{args.ticker} close price",
            ylabel="Price",
        )
        print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lstm-financial",
        description="LSTM forecasting for financial and macroeconomic time series",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="train on a synthetic series (no data files needed)")
    demo.add_argument("--periods", type=int, default=1000)
    _add_training_arguments(demo)
    demo.set_defaults(handler=command_demo)

    train = subparsers.add_parser("train", help="train on one ticker of the stock dataset")
    train.add_argument("--ticker", required=True)
    train.add_argument("--stock-file", default=None, help="override the stock CSV path")
    _add_training_arguments(train)
    train.set_defaults(handler=command_train)

    economic = subparsers.add_parser("economic", help="train on a macroeconomic indicator")
    economic.add_argument("--series-code", required=True, help="e.g. DSTKMKTXN")
    economic.add_argument("--country", default=None, help="ISO3 country code, e.g. GBR")
    economic.add_argument("--frequency", default="M", choices=["A", "Q", "M"])
    economic.add_argument("--economic-file", default=None)
    _add_training_arguments(economic)
    economic.set_defaults(handler=command_economic)

    explore = subparsers.add_parser("explore", help="summarise a dataset")
    explore.add_argument("dataset", choices=["stocks", "economic", "rbi"])
    explore.add_argument("--limit", type=int, default=25)
    explore.add_argument("--sheet", default="Quarterly", help="RBI workbook sheet")
    explore.add_argument("--stock-file", default=None)
    explore.add_argument("--economic-file", default=None)
    explore.add_argument("--rbi-file", default=None)
    explore.set_defaults(handler=command_explore)

    indicators = subparsers.add_parser("indicators", help="export technical indicators for a ticker")
    indicators.add_argument("--ticker", required=True)
    indicators.add_argument("--stock-file", default=None)
    indicators.add_argument("--output", default=None)
    indicators.add_argument("--artifacts-dir", default=None)
    indicators.add_argument("--plot", action="store_true")
    indicators.set_defaults(handler=command_indicators)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    pd.set_option("display.width", 140)
    try:
        return int(args.handler(args, settings))
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
