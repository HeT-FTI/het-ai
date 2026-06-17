"""Sphinx configuration for het-ai documentation."""

import os
import sys
from pathlib import Path

# -- Path setup ----------------------------------------------------------------
# Ensure het_ai package is importable for autodoc.
# The repo uses a src/ layout, so we add the src directory to sys.path.
_src_dir = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src_dir))

# -- Project information -------------------------------------------------------
project = "het-ai"
copyright = "2025, Shenzhen HeT Intelligent Control Co., Ltd."
author = "Chen Zhang"

# Dynamically read version from pyproject.toml
_repo_root = Path(__file__).resolve().parents[2]
_pyproject_path = _repo_root / "pyproject.toml"
_version = "1.0.0"
if _pyproject_path.exists():
    try:
        # tomllib is available in Python >= 3.11
        import tomllib
        with open(_pyproject_path, "rb") as f:
            _data = tomllib.load(f)
        _version = _data.get("project", {}).get("version", _version)
    except Exception:
        pass
release = _version
version = ".".join(release.split(".")[:2])  # e.g. "1.0"

# -- General configuration -----------------------------------------------------
extensions = [
    # Built-in Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    # Third-party extensions
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.kroki",
]

# Kroki settings (PlantUML via Kroki HTTP API — no local Java required)
kroki_server = "https://kroki.io"
kroki_diagram_types = ["plantuml"]
kroki_plantuml_output_format = "svg"

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}
autoclass_content = "class"

# Mock optional dependencies so autodoc can document dvc/mlflow modules
# without requiring the full platform extras to be installed.
autodoc_mock_imports = [
    "mlflow",
    "mlflow.data",
    "mlflow.onnx",
    "mlflow.pytorch",
    "mlflow.sklearn",
    "mlflow.pyfunc",
    "mlflow.keras",
    "dvc",
    "requests",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "Pillow",
    "joblib",
    "netCDF4",
    "arviz",
    "pymc",
]

# Autosummary settings
autosummary_generate = True

# Intersphinx mappings (link to external library docs)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# MyST parser settings (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

# Copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,4}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# General settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"
pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "het-ai Documentation"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93bbfd",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/HeT-FTI/het-ai",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/HeT-FTI/het-ai",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# -- Options for manual pages (optional) ---------------------------------------
# man_pages = [
#     ("index", "het-ai", "het-ai Documentation", [author], 1),
# ]

