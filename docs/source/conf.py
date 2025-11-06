#  noqa:D100
# Configuration file for the Sphinx documentation builder
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib
import sys

src_root = pathlib.Path("../..").resolve()
sys.path.insert(0, src_root)
project = "DTC Ice Sheets Development Guide"
company_copyright = "2025, Earthwave Ltd."
author = "Earthwave Ltd."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.intersphinx", "sphinx_copybutton"]
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = []

source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
