"""Templar Crusades - Training code efficiency crusades subnet."""

__version__ = "2.0.0"  # Major bump to test competition reset

# Competition version from major version number
# Major bump (2.x.x -> 3.x.x) = new competition (fresh start)
# Minor/patch bump (2.0.0 -> 2.1.0) = same competition continues
COMPETITION_VERSION: int = int(__version__.split(".")[0])

from crusades.logging import LOKI_URL, setup_loki_logger

__all__ = ["__version__", "COMPETITION_VERSION", "setup_loki_logger", "LOKI_URL"]
