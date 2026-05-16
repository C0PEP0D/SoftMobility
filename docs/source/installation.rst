============
Installation
============

Prerequisites
-------------

SoftMobility requires Python 3.10 or higher (dictated by the JAX dependency).
It also depends on:

- JAX (automatic differentiation and JIT compilation)
- NumPy (array operations)
- SciPy (scientific computing)
- Optax (gradient-based optimizers)
- matplotlib (plotting; used by tutorials and examples and rendered figures, including vector PDF export)

Installation from Source
------------------------

The package is not yet on PyPI. Clone the repository and install it in editable
mode inside an isolated environment. Two equivalent recipes are documented
below — use whichever matches your existing tooling.

With venv
~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/C0PEP0D/SoftMobility.git
    cd SoftMobility
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .

With conda
~~~~~~~~~~

If you prefer ``conda`` (or the faster, drop-in ``mamba``) for environment
management, the recommended pattern is to let conda manage the Python sandbox
and let pip install the package itself — JAX in particular is more reliable
when installed from PyPI than from conda-forge:

.. code-block:: bash

    git clone https://github.com/C0PEP0D/SoftMobility.git
    cd SoftMobility
    conda create -n softmobility python=3.11
    conda activate softmobility
    python -m pip install --upgrade pip
    python -m pip install -e .

Or, equivalently, use the bundled ``environment.yml``, which performs the same
steps in a single command. It must be run from the repository root because of
the ``-e .`` editable install:

.. code-block:: bash

    git clone https://github.com/C0PEP0D/SoftMobility.git
    cd SoftMobility
    conda env create -f environment.yml
    conda activate softmobility

Development and documentation tools
------------------------------------

Install the additional tools needed to run tests and build the documentation
(this works inside either a ``venv`` or a ``conda`` environment):

.. code-block:: bash

    pip install -r requirements-dev.txt

Verifying Installation
----------------------

.. code-block:: python

    import softmobility as sm
    print(f"SoftMobility version: {sm.__version__}")

.. _running-notebooks-on-google-colab:

Running tutorials and examples on Google Colab
----------------------------------------------

Notebooks in ``softmobility/tutorials`` (library introduction) and
``softmobility/examples`` (validation cases and case studies) can be run
directly in Google Colab — no local clone, fork, or install required.
Each notebook contains a first cell that installs ``SoftMobility`` from
GitHub when it detects a Colab runtime; locally the cell is a no-op.

To open a notebook in Colab, either click the corresponding badge in the
`README on GitHub <https://github.com/C0PEP0D/SoftMobility#try-the-notebooks-online>`_,
or build the URL by hand by replacing the GitHub URL prefix
``https://github.com/C0PEP0D/SoftMobility/blob/`` with
``https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/``.

Troubleshooting
---------------

If you encounter issues with JAX installation (especially on GPU systems), refer
to the `JAX installation guide <https://github.com/google/jax#installation>`_
for platform-specific instructions.

For macOS with Apple Silicon (M1/M2/M3), install the CPU build of JAX
explicitly:

.. code-block:: bash

    pip install "jax[cpu]"
