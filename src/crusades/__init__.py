"""Templar Crusades - Training code efficiency crusades subnet."""

__version__ = "0.7.0"

# Competition version from major.minor version number
# Major OR Minor bump = new competition (fresh start)
# Patch bump only = same competition continues
# Examples: "0.2.0" -> 2, "0.3.0" -> 3, "1.0.0" -> 100
_version_parts = __version__.split(".")
COMPETITION_VERSION: int = int(_version_parts[0]) * 100 + int(_version_parts[1])

try:
    from crusades.logging import LOKI_URL, setup_loki_logger
except ImportError:
    # During build-time version resolution, runtime deps may not be installed yet
    LOKI_URL = ""  # type: ignore[assignment]
    setup_loki_logger = None  # type: ignore[assignment]

__all__ = ["__version__", "COMPETITION_VERSION", "setup_loki_logger", "LOKI_URL"]
