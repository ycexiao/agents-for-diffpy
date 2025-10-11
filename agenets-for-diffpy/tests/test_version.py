"""Unit tests for __version__.py."""

import agenets_for_diffpy  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(agenets_for_diffpy, "__version__")
    assert agenets_for_diffpy.__version__ != "0.0.0"
