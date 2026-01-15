# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'satellite precoding techniques'
copyright = '2026, Wannes Baes'
author = 'Wannes Baes'
release = 'v0.1'


# -- Path setup --------------------------------------------------------------

import os
import sys

PROJECT_ROOT = os.path.abspath('../..')

# waterfilling module
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'su-mimo', 'waterfilling'))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

templates_path = [
    '_templates'
]

exclude_patterns = []

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'
html_static_path = ['_static']
