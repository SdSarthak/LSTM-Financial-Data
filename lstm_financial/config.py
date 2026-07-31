"""Runtime configuration.

Every path is resolved from environment variables so that no dataset location
is hardcoded in the source.  Copy ``.env.example`` to ``.env`` and adjust.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default file names of the datasets described in the README.  They are only
# used when the corresponding environment variable is not set.
DEFAULT_STOCK_FILE = "World-Stock-Prices-Dataset.csv"
DEFAULT_ECONOMIC_FILE = "2aaef3be-6f4c-4673-bc11-c6add6a8516a_Data.csv"
DEFAULT_ECONOMIC_METADATA_FILE = (
    "2aaef3be-6f4c-4673-bc11-c6add6a8516a_Series - Metadata.csv"
)
DEFAULT_RBI_FILE = "RBIB Table No. 01 _ Select Economic Indicators.xlsx"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def load_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """Populate ``os.environ`` from a simple ``KEY=value`` file.

    Keys already present in the environment win, so an explicit shell export
    always overrides the file.  Returns the values that were applied.
    """
    dotenv = Path(path) if path is not None else PROJECT_ROOT / ".env"
    applied: dict[str, str] = {}
    if not dotenv.is_file():
        return applied

    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


@dataclass(frozen=True)
class Settings:
    """Resolved locations for input data and generated artifacts."""

    data_dir: Path = field(default_factory=lambda: _env_path("LSTM_FIN_DATA_DIR", PROJECT_ROOT))
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("LSTM_FIN_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
    )
    stock_file: str = field(
        default_factory=lambda: os.environ.get("LSTM_FIN_STOCK_FILE", DEFAULT_STOCK_FILE)
    )
    economic_file: str = field(
        default_factory=lambda: os.environ.get("LSTM_FIN_ECONOMIC_FILE", DEFAULT_ECONOMIC_FILE)
    )
    economic_metadata_file: str = field(
        default_factory=lambda: os.environ.get(
            "LSTM_FIN_ECONOMIC_METADATA_FILE", DEFAULT_ECONOMIC_METADATA_FILE
        )
    )
    rbi_file: str = field(
        default_factory=lambda: os.environ.get("LSTM_FIN_RBI_FILE", DEFAULT_RBI_FILE)
    )
    random_seed: int = field(
        default_factory=lambda: int(os.environ.get("LSTM_FIN_SEED", "42"))
    )

    @property
    def stock_path(self) -> Path:
        return self.data_dir / self.stock_file

    @property
    def economic_path(self) -> Path:
        return self.data_dir / self.economic_file

    @property
    def economic_metadata_path(self) -> Path:
        return self.data_dir / self.economic_metadata_file

    @property
    def rbi_path(self) -> Path:
        return self.data_dir / self.rbi_file

    def ensure_artifacts_dir(self) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir


def get_settings(load_env_file: bool = True) -> Settings:
    """Build :class:`Settings` from the environment (reading ``.env`` first)."""
    if load_env_file:
        load_dotenv()
    return Settings()
