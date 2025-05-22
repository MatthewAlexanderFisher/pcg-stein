import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "pcg-stein"
copyright = "2025, Matthew Fisher"
author = "Matthew Fisher"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx_autodoc_typehints",  # autogenerate type hints
    "sphinx.ext.mathjax",  # renders LaTeX
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__, __call__",
    "show-inheritance": True,
}


autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]