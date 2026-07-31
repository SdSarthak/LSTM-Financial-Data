"""Entry point for the LSTM financial forecasting pipeline.

Running this file without arguments trains the synthetic demo, which needs no
dataset.  Every other command lives behind the same CLI:

    python main.py train --ticker AAPL
    python main.py explore stocks
    python main.py --help
"""

from __future__ import annotations

import sys

from lstm_financial.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:] or ["demo"]
    raise SystemExit(main(arguments))
