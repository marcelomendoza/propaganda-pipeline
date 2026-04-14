How to Install
==============

Requirements
------------

Python **3.10** or later is required.

Step 1 — Install dependencies
------------------------------

Install all required dependencies using the provided ``requirements.txt``:

.. code-block:: bash

    pip install -r requirements.txt

The ``requirements.txt`` file contains the following packages:

.. code-block:: text

    dspy>=2.6
    pydantic>=2.9
    ipython>=8.7
    openai>=1.0

Step 2 — Install Propaganda Pipeline
--------------------------------------

Once the dependencies are in place, install the library from GitHub:

.. code-block:: bash

    pip install git+https://github.com/marcelomendoza/propaganda-pipeline.git@v0.1.0-beta
