# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import _comparison_generator  # NOQA

# Generate comparison table.
with open('comparison_table.rst.inc', 'w') as f:
    f.write(_comparison_generator.generate())

# -- Project information -----------------------------------------------------

project = 'nlcpy'
copyright = '2020, NEC Corporation'
author = 'NEC'

here = os.path.abspath(os.path.dirname(__file__))
exec(open(here + '/../../nlcpy/_version.py').read())
version = __version__  # NOQA
release = __version__  # NOQA

# -- General configuration ---------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

html_show_sourcelink = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'pydata_sphinx_theme'

# Custom sidebar templates, maps document names to template names.
html_sidebars = {'**': ['relations.html', ]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_style = 'css/custom.css'

html_context = {
    # 'css_files': [
    #     'https://media.readthedocs.org/css/sphinx_rtd_theme.css',
    #     'https://media.readthedocs.org/css/readthedocs-doc-embed.css',
    #     '_static/css/custom.css',
    # ],
    'support_languages': {
        'en': 'English',
        'ja': 'Japanese',
    },
}

html_logo = '../image/NLCPy_banner.png'

# html_theme_options = {
#     'navigation_depth': 4,
# }

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Extension configuration -------------------------------------------------

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              'sphinx.ext.extlinks',
              'sphinx.ext.intersphinx',
              # 'sphinx.ext.mathjax',
              'sphinx.ext.imgmath',
              # 'sphinx.ext.napoleon',
              # 'numpydoc',
              'nlcpydoc',
              'sphinx.ext.viewcode',
              'matplotlib.sphinxext.plot_directive',
              ]

autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Napoleon settings
# napoleon_use_ivar = True
# napoleon_include_special_with_doc = True
# napoleon_use_param = True
# napoleon_use_rtype = False

# nlcpydoc settings
nlcpydoc_use_plots = True
nlcpydoc_attributes_as_param_list = False
nlcpydoc_show_class_members = False

# matplotlib plot directive
plot_include_source = True
plot_formats = [("png", 60)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_pre_code = """
import numpy as np
import nlcpy as vp
import matplotlib.pyplot as plt
"""

gettext_compact = False
locale_dirs = ['locale/']


doctest_global_setup = '''
'''
doctest_test_doctest_blocks = 'default'
doctest_default_flags = 0

#doctest_default_flags = (0
#    | doctest.DONT_ACCEPT_TRUE_FOR_1
#    | doctest.ELLIPSIS
#    | doctest.IGNORE_EXCEPTION_DETAIL
#    | doctest.NORMALIZE_WHITESPACE
#)

imgmath_image_format = 'svg'
imgmath_font_size = 16
