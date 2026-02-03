"""Templar Crusades - Training code efficiency crusades subnet."""

__version__ = "3.0.0"  # Major bump to test competition reset

from crusades.logging import setup_loki_logger, LOKI_URL


def get_competition_version() -> int:
    """Get competition version from major version number.

    The major version determines competition boundaries:
    - Major bump (2.x.x -> 3.x.x) = new competition (fresh start)
    - Minor/patch bump (2.0.0 -> 2.1.0) = same competition continues

    Returns:
        Major version number as integer (e.g., 2 for "2.0.0")
    """
    return int(__version__.split(".")[0])


__all__ = ["__version__", "get_competition_version", "setup_loki_logger", "LOKI_URL"]
