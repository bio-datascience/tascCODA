# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
import datetime
import matplotlib

from pathlib import Path

matplotlib.use('agg')

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / '_ext')]

import tasccoda
on_rtd = os.environ.get('READTHEDOCS') == 'True'

needs_sphinx = "2.0"

# -- Retrieve notebooks ------------------------------------------------

from urllib.request import urlretrieve

notebooks_url = "https://github.com/bio-datascience/tascCODA/tree/main/tutorials"
notebooks = [
    "tascCODA_tutorial.ipynb",
]
for nb in notebooks:
    try:
        urlretrieve(notebooks_url + nb, nb)
    except:
        pass

# -- Project information -----------------------------------------------------

project = 'tascCODA'
title = 'tascCODA: Bayesian Tree-Aggregated Analysis of Compositional Amplicon and Single-Cell Data'
author = 'Johannes Ostner, Salomé Carcy, Christian L. Müller'
copyright = f"{datetime.datetime.now():%Y}, {author}"

version = tasccoda.__version__.replace(".dirty", "")
release = version

# -- General configuration ---------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = [".rst", ".ipynb"]
master_doc = 'index'
default_role = 'literal'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.doctest',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              "sphinx_autodoc_typehints",
              "nbsphinx",
              "scanpydoc",
              *[p.stem for p in (HERE / 'extensions').glob('*.py')],
              ]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False

intersphinx_mapping = dict(
    python=("https://docs.python.org/3", None),
    anndata=("https://anndata.readthedocs.io/en/latest/", None),
    scanpy=("https://scanpy.readthedocs.io/en/latest/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    matplotlib=('https://matplotlib.org/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    seaborn=('https://seaborn.pydata.org/', None),

)

# Add notebooks prolog to Google Colab and nbviewer
nbsphinx_prolog = r"""
{% set docname = 'bio-datascience/tascCODA/blob/main' + env.doc2path(env.docname, base=None) %}
.. raw:: html

    <div class="note">
      <a href="https://colab.research.google.com/{{ docname|e }}" target="_parent">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
      <a href="https://nbviewer.jupyter.org/{{ docname|e }}" target="_parent">
      <img src="https://github.com/theislab/scCODA/raw/master/docs/source/_static/nbviewer_badge.svg" alt="Open In nbviewer"/></a>
    </div>
"""


# -- Options for HTML output -------------------------------------------------

html_theme = 'scanpydoc'
html_theme_options = dict(navigation_depth=1, titles_only=True)
github_repo = "tasccoda"
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user='bio-datascience',  # Username
    github_repo='tascCODA',  # Repo name
    github_version='main',  # Version
    conf_py_path='/docs/',  # Path in the checkout to the docs root
)
html_static_path = ['_static']
html_show_sphinx = False

# def setup(app):
#     app.warningiserror = on_rtd

# -- Options for other output ------------------------------------------

htmlhelp_basename = "tascCODAdoc"
title_doc = f"{project} documentation"

# latex_engine = "xelatex"
latex_elements = {
'preamble': r'''
\usepackage[utf8]{inputenc}

\newcommand{\block}[1]{\raisebox{\dimexpr(\fontcharht\font`X-1em)/2}{\rule{1em}{#1\dimexpr1em/8}}}

\DeclareUnicodeCharacter{2581}{\block{1}}
\DeclareUnicodeCharacter{2582}{\block{2}}
\DeclareUnicodeCharacter{2583}{\block{3}}
\DeclareUnicodeCharacter{2584}{\block{4}}
\DeclareUnicodeCharacter{2585}{\block{5}}
\DeclareUnicodeCharacter{2586}{\block{6}}
\DeclareUnicodeCharacter{2587}{\block{7}}
\DeclareUnicodeCharacter{2588}{\block{8}}
'''
}

latex_documents = [(master_doc, f"{project}.tex", title_doc, author, "manual")]
man_pages = [(master_doc, project, title_doc, [author], 1)]
texinfo_documents = [
    (master_doc, project, title_doc, author, project, title, "Miscellaneous")
]

# -- Override some classnames in autodoc --------------------------------------------

qualname_overrides = {
}
