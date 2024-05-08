"""Sphinx configuration."""

project = "pyXcell"
author = "Fan Zhang Lab"
copyright = "2024, Fan Zhang Lab"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
