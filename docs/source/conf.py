"""
Sphinx configuration for code documentation.

This file controls:
- What modules Sphinx can import (sys.path)
- Which Sphinx extensions are enabled (autodoc, napoleon, MyST, ...)
- How API docs are generated and ordered (autosummary, autodoc_member_order, ...)
- HTML output settings (theme, static files, templates)
"""

# -- Project information -----------------------------------------------------

project = 'satellite precoding techniques'
copyright = '2026, Wannes Baes'
author = 'Wannes Baes'
release = 'v0.1'


# -- Path setup --------------------------------------------------------------
# Make sure Sphinx can import your project packages/modules when building docs.
# This is especially important when your code is not installed as a site-package.

import os
import sys

# Absolute path to the repository root.
PROJECT_ROOT = os.path.abspath('../..')

# Add import paths so autodoc can find modules referenced in .rst files.
# - PROJECT_ROOT: for top-level imports (if any)
# - su-mimo/src and mu-mimo/src: where the actual Python packages live
sys.path.insert(0, os.path.join(PROJECT_ROOT))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'su-mimo', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'mu-mimo', 'src'))


# -- General configuration ---------------------------------------------------

# Sphinx extensions used by this documentation:
# - autodoc: pull docstrings from Python code into the docs
# - napoleon: support Google/Numpy style docstrings
# - mathjax: render LaTeX math in HTML
# - viewcode: add links to highlighted source code
# - autosummary: generate summary tables/pages for documented objects
# - myst_parser: allow Markdown (.md) sources alongside reStructuredText (.rst)
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

# Enable specific MyST (Markdown) extensions.
# - amsmath: AMS math environments
# - dollarmath: $...$ / $$...$$ math
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# Paths that contain templates, relative to the source directory.
templates_path = [
    '_templates'
]

# Static assets (images, CSS, JS). Referenced via /_static/...
html_static_path = [
    '_static'
]

# Patterns (relative to source dir) to ignore when scanning for sources.
exclude_patterns = []

# Autosummary: automatically generate stub pages/files where applicable.
autosummary_generate = True

# Do not inject every autodoc object (methods/attributes) into the page TOC.
toc_object_entries = False

# Do not prefix documented objects with their module name in headings/signatures.
add_module_names = False

# Order class members in the docs as they appear in the Python source file, instead of alphabetical.
autodoc_member_order = "bysource"


# -- HTML output -------------------------------------------------------------

# Chosen HTML theme for generated documentation.
html_theme = 'bizstyle'