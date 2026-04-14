import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

project = "Propaganda Pipeline"
author = "Marcelo Mendoza"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_logo = "propaganda_logotipo.png"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-sidebar-background": "#ffffff",
        "color-background-primary": "#eaf6ff",
        "color-background-secondary": "#eaf6ff",
    },
    "footer_icons": [],
}
html_last_updated_fmt = None
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "sidebar/mit-badge.html",
    ]
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = ["dspy", "openai", "pydantic"]

suppress_warnings = ["docutils"]

nbsphinx_execute = "never"

# Load pre-translated docstrings cache
_CACHE_FILE = Path(__file__).parent / "translations_cache.json"
_translations: dict = {}
if _CACHE_FILE.exists():
    _translations = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))


def _replace_docstring(app, what, name, obj, options, lines):
    """Replace Spanish docstrings with cached English translations."""
    short_name = name.split(".")[-1]
    if short_name in _translations:
        translated = _translations[short_name]
        lines[:] = translated.splitlines()


def setup(app):
    app.connect("autodoc-process-docstring", _replace_docstring)
