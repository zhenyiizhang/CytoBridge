import os
import sys
from pathlib import Path
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


project = 'CytoBridge'
copyright = '2025, CytoBridge Developer team'
author = 'CytoBridge Developer team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",                    # 让 notebook 直接当文档
    "sphinx_autodoc_typehints",
]
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
nb_execution_mode = "off"
templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# conf.py 片段
html_static_path = ['_static']


html_logo="_static/logo.png"



