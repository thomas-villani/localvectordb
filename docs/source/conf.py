# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "localvectordb"
copyright = "2025-2026, Tom Villani"
author = "Tom Villani, Ph.D."
release = "0.1.0rc1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

# Builtins used as annotations (e.g. `valid_types() -> Tuple[type, ...]`) have no
# target in this project, so Sphinx fuzzy-matches them to whatever looks close --
# bare `type` was resolving to MetadataField.type, a silently wrong link. Give the
# builtins a real inventory instead.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
intersphinx_timeout = 15

templates_path = ["_templates"]
exclude_patterns = []

# Do NOT add "inherited-members": False here. Sphinx's process_documenter_options
# overwrites a directive option with the config default whenever that default is
# not a string, so the entry silently clobbered the `:inherited-members:` on
# modules/localvectordb.database.rst -- and LocalVectorDB, whose every method
# comes from a mixin, documented zero members while appearing to build fine.
# Omitting the key gives the same off-by-default behaviour without the clobber.
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "special-members": "__init__",
    "show-inheritance": True,
}
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "signature"

# Render a numpydoc "Attributes" section as an :ivar: field list rather than as
# standalone `.. attribute::` directives. Without this, every documented
# dataclass field is described twice on its own page -- once by napoleon from
# the docstring, once by autodoc's undoc-members from the annotation -- which
# Sphinx reports as a duplicate object description.
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
