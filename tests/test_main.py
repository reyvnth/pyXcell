"""Test cases for the __main__ module."""

from pyXcell import __version__


def test_version():
    """Version assert!"""
    assert __version__ == "0.0.1"
