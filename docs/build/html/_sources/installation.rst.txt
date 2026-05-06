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
- Plotly (plotting; used by tutorials and rendered figures)

Installation from Source
------------------------

The package is not yet on PyPI. Clone the repository and install in editable
mode:

.. code-block:: bash

    git clone https://github.com/celoy/SoftMobility.git
    cd SoftMobility
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .

Development and documentation tools
------------------------------------

Install the additional tools needed to run tests and build the documentation:

.. code-block:: bash

    pip install -r requirements-dev.txt

Verifying Installation
----------------------

.. code-block:: python

    import softmobility as sm
    print(f"SoftMobility version: {sm.__version__}")

Troubleshooting
---------------

If you encounter issues with JAX installation (especially on GPU systems), refer
to the `JAX installation guide <https://github.com/google/jax#installation>`_
for platform-specific instructions.

For macOS with Apple Silicon (M1/M2/M3), install the CPU build of JAX
explicitly:

.. code-block:: bash

    pip install "jax[cpu]"
