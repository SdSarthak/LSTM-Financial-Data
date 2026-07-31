import pandas as pd
import pytest

from lstm_financial import cli, config, plots


def test_load_dotenv_does_not_override_the_environment(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text('LSTM_FIN_SEED=99\nLSTM_FIN_DATA_DIR="/data/finance"\n# comment\n\n', encoding="utf-8")
    monkeypatch.delenv("LSTM_FIN_SEED", raising=False)
    monkeypatch.setenv("LSTM_FIN_DATA_DIR", "/already/set")

    applied = config.load_dotenv(dotenv)
    assert applied == {"LSTM_FIN_SEED": "99"}
    assert config.Settings().random_seed == 99
    assert str(config.Settings().data_dir) in {"/already/set", "\\already\\set"}


def test_settings_paths_follow_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LSTM_FIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LSTM_FIN_STOCK_FILE", "prices.csv")
    monkeypatch.setenv("LSTM_FIN_ARTIFACTS_DIR", str(tmp_path / "out"))
    settings = config.Settings()
    assert settings.stock_path == tmp_path / "prices.csv"
    assert settings.ensure_artifacts_dir().is_dir()


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert config.load_dotenv(tmp_path / "absent.env") == {}


def test_parser_exposes_every_command():
    parser = cli.build_parser()
    for command in ["demo", "train", "economic", "explore", "indicators"]:
        assert parser.parse_args([command, *_minimal_arguments(command)]).command == command


def _minimal_arguments(command: str) -> list[str]:
    return {
        "demo": [],
        "train": ["--ticker", "AAA"],
        "economic": ["--series-code", "CORENS"],
        "explore": ["stocks"],
        "indicators": ["--ticker", "AAA"],
    }[command]


def test_training_arguments_are_forwarded():
    args = cli.build_parser().parse_args(
        ["demo", "--lookback", "5", "--units", "8", "4", "--epochs", "3", "--quiet"]
    )
    kwargs = cli._training_kwargs(args)
    assert kwargs["lookback"] == 5
    assert kwargs["units"] == (8, 4)
    assert kwargs["epochs"] == 3
    assert kwargs["verbose"] == 0


def test_explore_stocks_lists_tickers(stock_csv, monkeypatch, capsys):
    monkeypatch.setenv("LSTM_FIN_DATA_DIR", str(stock_csv.parent))
    monkeypatch.setenv("LSTM_FIN_STOCK_FILE", stock_csv.name)
    assert cli.main(["explore", "stocks"]) == 0
    output = capsys.readouterr().out
    assert "2 tickers" in output
    assert "AAA" in output and "BBB" in output


def test_explore_economic_summarises_series(economic_csv, monkeypatch, capsys):
    monkeypatch.setenv("LSTM_FIN_DATA_DIR", str(economic_csv.parent))
    monkeypatch.setenv("LSTM_FIN_ECONOMIC_FILE", economic_csv.name)
    assert cli.main(["explore", "economic"]) == 0
    output = capsys.readouterr().out
    assert "DSTKMKTXN" in output
    assert "2 countries" in output


def test_explore_rbi_summarises_indicators(rbi_workbook, monkeypatch, capsys):
    monkeypatch.setenv("LSTM_FIN_DATA_DIR", str(rbi_workbook.parent))
    monkeypatch.setenv("LSTM_FIN_RBI_FILE", rbi_workbook.name)
    assert cli.main(["explore", "rbi", "--sheet", "Quarterly"]) == 0
    output = capsys.readouterr().out
    assert "gva_at_basic_prices" in output
    assert "2024-25 Q4" in output


def test_indicators_command_writes_a_csv(stock_csv, tmp_path, monkeypatch, capsys):
    prices = pd.read_csv(stock_csv)
    # 12 rows is shorter than the indicator warm-up, so lengthen the fixture here.
    repeated = pd.concat([prices] * 6, ignore_index=True)
    repeated["Date"] = [
        f"{d:%Y-%m-%d} 00:00:00-05:00" for d in pd.bdate_range("2024-01-01", periods=len(repeated))
    ]
    repeated["Ticker"] = "AAA"
    repeated["Close"] = range(100, 100 + len(repeated))
    long_csv = tmp_path / "long.csv"
    repeated.to_csv(long_csv, index=False)

    output = tmp_path / "indicators.csv"
    assert cli.main(["indicators", "--ticker", "AAA", "--stock-file", str(long_csv),
                     "--output", str(output), "--artifacts-dir", str(tmp_path)]) == 0
    assert output.is_file()
    written = pd.read_csv(output)
    assert {"close", "rsi_14", "macd"} <= set(written.columns)
    assert "wrote" in capsys.readouterr().out


def test_missing_dataset_returns_exit_code_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LSTM_FIN_DATA_DIR", str(tmp_path))
    assert cli.main(["explore", "stocks"]) == 1
    assert "error:" in capsys.readouterr().err


def test_plot_helpers_write_files(tmp_path, price_frame):
    series_path = plots.plot_series(price_frame["close"], tmp_path / "series.png")
    prediction_path = plots.plot_predictions([1.0, 2.0], [1.1, 1.9], tmp_path / "pred.png")
    history_path = plots.plot_training_history(
        {"loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}, tmp_path / "history.png"
    )
    forecast_path = plots.plot_forecast([1.0, 2.0], [2.1, 2.2], tmp_path / "forecast.png")
    for path in (series_path, prediction_path, history_path, forecast_path):
        assert path.is_file() and path.stat().st_size > 0
