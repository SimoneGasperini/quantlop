"""Sphinx configuration for the quantlop documentation."""

from importlib.metadata import version as package_version


project = "quantlop"
author = "Simone Gasperini"
release = package_version("quantlop")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_design",
]

autodoc_default_options = {
    "members": True,
}
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = f"quantlop {release}"
html_theme_options = {
    "github_url": "https://github.com/SimoneGasperini/quantlop",
    "logo": {
        "alt_text": "quantlop documentation - Home",
        "image_light": "../light_logo.png",
        "image_dark": "../dark_logo.png",
    },
    "show_toc_level": 2,
}
