===========
Installation
===========

Prerequisites
-------------

SoftMobility requires Python 3.8 or higher. It also depends on several scientific computing libraries:

- JAX (for automatic differentiation and GPU acceleration)
- NumPy (for numerical computations)
- SciPy (for scientific computing)
- Matplotlib (for plotting)
- Plotly (for interactive visualizations)

Installation from PyPI
----------------------

The easiest way to install SoftMobility is using pip:

.. code-block:: bash

    pip install softmobility

Installation from Source
------------------------

To install from source, first clone the repository:

.. code-block:: bash

    git clone https://github.com/celoy/SoftMobility.git
    cd SoftMobility

Then install in development mode:

.. code-block:: bash

    pip install -e .

For development with all optional dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

Optional Dependencies
---------------------

For development and testing, install the optional dependencies:

.. code-block:: bash

    pip install -e ".[test]"

Verifying Installation
---------------------

To verify that SoftMobility is installed correctly, you can run:

.. code-block:: python

    import softmobility as sm
    print(f"SoftMobility version: {sm.__version__}")

Troubleshooting
---------------

If you encounter issues with JAX installation (especially on GPU systems), refer to the `JAX installation guide <https://github.com/google/jax#installation>`_ for platform-specific instructions.

For MacOS with Apple Silicon (M1/M2), you may need to install JAX with:

.. code-block:: bash

    pip install "jax[cpu]"