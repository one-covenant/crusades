"""Templar Crusades - Training code efficiency crusades subnet."""

__version__ = "2.0.0"  # Major bump to test competition reset

# Competition version from major.minor version number
# Major OR Minor bump = new competition (fresh start)
# Patch bump only = same competition continues
# Examples: "2.0.0" → 200, "2.1.0" → 201, "3.0.0" → 300
_version_parts = __version__.split(".")
COMPETITION_VERSION: int = int(_version_parts[0]) * 100 + int(_version_parts[1])

from crusades.logging import LOKI_URL, setup_loki_logger

__all__ = ["__version__", "COMPETITION_VERSION", "setup_loki_logger", "LOKI_URL"]
