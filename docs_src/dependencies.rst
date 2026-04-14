Dependencies
============

Propaganda Pipeline relies on the following third-party libraries:

Runtime
-------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Package
     - Version
     - Purpose
   * - `dspy <https://github.com/stanfordnlp/dspy>`_
     - >= 2.6
     - LLM programming framework used to define and run detection signatures and modules.
   * - `pydantic <https://docs.pydantic.dev>`_
     - >= 2.9
     - Data validation and structured output schemas for detection results.
   * - `openai <https://github.com/openai/openai-python>`_
     - >= 1.0
     - OpenAI API client used as the default LLM backend.
   * - `ipython <https://ipython.org>`_
     - >= 8.7
     - Interactive display utilities used in notebook environments.

Development / Documentation
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Package
     - Version
     - Purpose
   * - `sphinx <https://www.sphinx-doc.org>`_
     - >= 8
     - Documentation generator.
   * - `furo <https://pradyunsg.me/furo>`_
     - latest
     - Sphinx HTML theme.
   * - `nbsphinx <https://nbsphinx.readthedocs.io>`_
     - latest
     - Renders Jupyter notebooks as Sphinx pages.
   * - `jupyter <https://jupyter.org>`_
     - latest
     - Notebook environment for the usage tutorial.

Python Requirement
------------------

Python **3.10** or later is required.
