"""Sphinx configuration for the quantlop documentation."""

from importlib.metadata import version as package_version


project = "quantlop"
author = "Simone Gasperini"
release = package_version("quantlop")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_design",
]

autodoc_default_options = {
    "members": True,
}
autodoc_member_order = "bysource"
numpydoc_attributes_as_param_list = True
numpydoc_class_members_toctree = False

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = f"quantlop {release}"
html_theme_options = {
    "github_url": "https://github.com/SimoneGasperini/quantlop",
    "logo": {
        "alt_text": "quantlop documentation - Home",
        "image_light": "../assets/light_logo.png",
        "image_dark": "../assets/dark_logo.png",
    },
    "show_toc_level": 2,
}
