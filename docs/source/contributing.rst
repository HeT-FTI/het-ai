Contributing
============

Thank you for your interest in contributing to het-ai! This guide covers
setting up your development environment, running tests, and building the
documentation.

Development Environment
-----------------------

1. **Clone the repository:**

   .. code-block:: bash

       git clone https://github.com/HeT-FTI/het-ai.git
       cd het-ai

2. **Create a virtual environment and install in editable mode:**

   .. code-block:: bash

       python -m venv .venv
       # Windows
       .venv\Scripts\activate
       # Linux / macOS
       source .venv/bin/activate

       pip install -e ".[dev]"

3. **Install framework extras as needed for tests:**

   .. code-block:: bash

       pip install -e ".[examples]"

Running Tests
-------------

The test suite uses ``pytest``. All 12 example cases are parametrised through
``tests/test_simulate_all.py`` which runs ``dry_run()`` on each case:

.. code-block:: bash

    # Run all tests
    python -m pytest tests/test_simulate_all.py -v

    # Run a specific test
    python -m pytest tests/test_simulate_all.py -k "case1" -v

    # Run with coverage (if installed)
    pip install pytest-cov
    python -m pytest tests/test_simulate_all.py --cov=het_ai --cov-report=term

Code Style
----------

- Follow `PEP 8 <https://peps.python.org/pep-0008/>`_.
- Docstrings are written in **Chinese** (project convention). Use Google-style
  formatting.
- Keep imports organised: standard library → third-party → local.
- Use type hints for all public methods and functions.

Adding a New Case
-----------------

To contribute a new example case:

1. Create a new file ``tests/cases/caseNN_descriptive_name.py`` following the
   naming convention.
2. Subclass :class:`~het_ai.studio.base.BaseTrainer` and implement:
   - ``load_data()``
   - ``mock_data()``
   - ``train()`` (decorated with ``@BaseTrainer.search(...)``)
   - ``export_model()`` (if needed)
3. Add the new case to ``tests/test_simulate_all.py`` in the parametrised test:

   .. code-block:: python

       @pytest.mark.parametrize("case_module", [
           ...,
           "tests.cases.caseNN_descriptive_name",
       ])
       def test_each_case_dry_run(case_module):
           ...

4. Ensure the case passes dry-run in under 30 seconds:

   .. code-block:: bash

       python -m pytest tests/test_simulate_all.py -k "caseNN" -v

Documentation
-------------

Documentation is built with **Sphinx** using the ``furo`` theme and is hosted on
`Read the Docs <https://readthedocs.org/>`_.

**Build locally:**

.. code-block:: bash

    pip install -e ".[docs]"
    cd docs
    sphinx-build -b html source build/html -W

**Open in browser:**

.. code-block:: bash

    # On Windows
    start docs/build/html/index.html
    # On macOS
    open docs/build/html/index.html
    # On Linux
    xdg-open docs/build/html/index.html

**Documentation structure:**

- ``docs/source/guides/`` — user guides and conceptual documentation
- ``docs/source/api/`` — auto-generated API reference (from docstrings)
- ``docs/source/examples/`` — example walkthroughs

Build & Release
---------------

.. code-block:: bash

    # Build distribution
    pip install build
    python -m build

    # Upload to PyPI (maintainers only)
    pip install twine
    python -m twine upload dist/*

