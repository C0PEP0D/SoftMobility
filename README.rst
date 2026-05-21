============
SoftMobility
============

.. image:: https://github.com/C0PEP0D/SoftMobility/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/C0PEP0D/SoftMobility/actions/workflows/testing.yml
   :alt: Test status

SoftMobility is a Python library for modelling deformable assemblies of
spheres in Stokes flows. It is intended for scientific users who want to define
soft bodies, compute mobility tensors, run differentiable simulations, and
optimize design parameters with JAX.

The package is imported as in Python as ``softmobility``.

Try the notebooks online
------------------------

The fastest way to get a feel for SoftMobility is to run a notebook
directly in your browser via Google Colab — no clone, fork, or local
install required. The first cell of every notebook installs
``SoftMobility`` from this repository when it detects a Colab runtime
(locally the cell is a no-op).

Click a badge below to launch the corresponding notebook in Colab.

Tutorials (library introduction):

* |colab-t01| ``01_assembly_creation``
* |colab-t02| ``02_rigid_mobility``
* |colab-t03| ``03_soft_mobility_simulation``
* |colab-t04| ``04_optimization``
* |colab-t05| ``05_figure_styling``

Examples (validation cases & case studies):

* |colab-e01| ``01_sinking_rigid_body``
* |colab-e02| ``02_sinking_fiber``
* |colab-e03| ``03_rotating_fiber``
* |colab-e04| ``04_fiber_in_shear``
* |colab-e05| ``05_jeffery_rigid``
* |colab-e06| ``06_jeffery_soft``
* |colab-e07| ``07_three_sphere_swimmer``
* |colab-e08| ``08_soft_surfer``

For any notebook not listed above, you can build a Colab URL by hand by
replacing the GitHub URL prefix
``https://github.com/C0PEP0D/SoftMobility/blob/`` with
``https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/``.

Installation
------------

SoftMobility requires Python 3.10 or newer. For a new user, the safest path
is to work in an isolated environment. The Sphinx ``installation`` page in
the documentation contains the same recipes plus troubleshooting notes
(Apple Silicon, GPU JAX builds, etc.).

From PyPI
~~~~~~~~~

Recommended for most users. Inside an isolated environment of your choice:

.. code-block:: bash

   python -m pip install softmobility

If you also want to run the bundled tutorials and examples or to modify the
library, install from source instead — recipes below.

With venv (from source)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/C0PEP0D/SoftMobility.git
   cd SoftMobility
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

With conda (from source)
~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer ``conda`` (or the faster, drop-in ``mamba``) for environment
management, the recommended pattern is to let conda manage the Python sandbox
and let pip install the package itself — JAX and a few other dependencies
install more reliably from PyPI than from conda-forge:

.. code-block:: bash

   git clone https://github.com/C0PEP0D/SoftMobility.git
   cd SoftMobility
   conda create -n softmobility python=3.11
   conda activate softmobility
   python -m pip install --upgrade pip
   python -m pip install -e .

Or, equivalently, use the bundled ``environment.yml`` (which performs the same
steps in a single command and must be run from the repository root because of
the ``-e .`` editable install):

.. code-block:: bash

   git clone https://github.com/C0PEP0D/SoftMobility.git
   cd SoftMobility
   conda env create -f environment.yml
   conda activate softmobility

Verify the installation:

.. code-block:: bash

   python -c "import softmobility as sm; print(sm.__version__)"

Quick Start
-----------

Start with the built-in input and flow objects:

.. code-block:: python

   import jax.numpy as jnp
   import softmobility as sm

   gravity = sm.gravity_field(g=9.81)
   flow = sm.shear_flow(shear_rate=1.0)

   pos = jnp.array([0.0, 2.0, 0.0])
   print(gravity.vector(pos))      # [0, 0, -9.81]
   print(flow.velocity(pos))       # [2, 0, 0]
   print(flow.gradient(pos))       # velocity-gradient matrix

A complete simulation uses three pieces:

1. ``sm.SoftBody`` for the deformable sphere assembly.
2. ``sm.Flow`` and optional ``sm.Field`` or ``sm.Scalar`` inputs.
3. ``sm.FlowBodyRollout`` to integrate the body trajectory.

.. code-block:: python

   import jax.numpy as jnp
   import softmobility as sm

   yaml_text = """
   dof_names: [x]
   design_names: [radius, length, k]
   defaults:
     x0: 0.1
     radius: 0.25
     length: 1.0
     k: 1.0
   spheres:
     - radius: radius
       position: [-length / 2, 0, 0]
       orientation: [0, x0, 0]
       torque: [0, -k * x0, 0]
     - radius: radius
       position: [length / 2, 0, 0]
       orientation: [0, -x0, 0]
       torque: [0, k * x0, 0]
   """

   body = sm.SoftBody(yaml_text, verbose=False)
   rollout = sm.FlowBodyRollout(body, sm.no_flow())

   positions, orientations, dofs = rollout.rollout(
       dt=0.01,
       n_steps=100,
       init_position=jnp.zeros(3),
       init_orientation=jnp.zeros(3),
   )

Tutorials and examples
----------------------

The notebooks ship in two folders. ``softmobility/tutorials`` contains
pedagogical walk-throughs of the API, while ``softmobility/examples``
collects validation cases against published results and original case
studies. New users should start with ``tutorials/01_assembly_creation``
and work through the tutorials before moving on to the examples.

**Tutorials** (``softmobility/tutorials/``)

* ``01_assembly_creation.ipynb`` — methods to create a sphere assembly
* ``02_rigid_mobility.ipynb`` — mobility properties of a rigid sphere assembly
* ``03_soft_mobility_simulation.ipynb`` — soft mobility tensors and simulation of a trajectory
* ``04_optimization.ipynb`` — optimization principles
* ``05_figure_styling.ipynb`` — paper-figure aesthetics with ``figstyle``

**Examples** (``softmobility/examples/``)

* ``01_sinking_rigid_body.ipynb`` — sinking trajectory of a rigid body
* ``02_sinking_fiber.ipynb`` — settling flexible fiber
* ``03_rotating_fiber.ipynb`` — rotating elastic fiber
* ``04_fiber_in_shear.ipynb`` — flexible fiber with intrinsic curvature in shear flow 
* ``05_jeffery_rigid.ipynb`` — Jeffery orbits of a rigid dumbbell
* ``06_jeffery_soft.ipynb`` — Jeffery orbit of an elastic dumbbell
* ``07_three_sphere_swimmer.ipynb`` — three-sphere swimmer with a passive elastic arm 
* ``08_soft_surfer.ipynb`` — soft surfer in Taylor-Green vortices

Build the documentation locally
-------------------------------

The Sphinx sources live in ``docs/source``. To build the HTML pages, first
install the development tools (which include Sphinx and its extensions):

.. code-block:: bash

   pip install -r requirements-dev.txt

Then build from the repository root:

.. code-block:: bash

   make -C docs html
   open docs/build/html/index.html        # macOS
   xdg-open docs/build/html/index.html    # Linux

The strict, warnings-as-errors build that mirrors CI is documented in the
*Developers* page (``docs/source/developers.rst``).

Contributing
------------

Contributions are welcome. The *Developers* page in the documentation
(``docs/source/developers.rst``) covers the development install, running the
tests, building the docs strictly, the versioning and release process, and
the policy for opening issues and pull requests. Please read it before
sending a PR.

License
-------

SoftMobility is distributed under the 3-clause BSD license. See ``LICENSE``
for details.

.. |colab-t01| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/01_assembly_creation.ipynb
.. |colab-t02| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/02_rigid_mobility.ipynb
.. |colab-t03| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/03_soft_mobility_simulation.ipynb
.. |colab-t04| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/04_optimization.ipynb
.. |colab-t05| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/05_figure_styling.ipynb
.. |colab-e01| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/01_sinking_rigid_body.ipynb
.. |colab-e02| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/02_sinking_fiber.ipynb
.. |colab-e03| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/03_rotating_fiber.ipynb
.. |colab-e04| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/04_fiber_in_shear.ipynb
.. |colab-e05| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/05_jeffery_rigid.ipynb
.. |colab-e06| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/06_jeffery_soft.ipynb
.. |colab-e07| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/07_three_sphere_swimmer.ipynb
.. |colab-e08| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/examples/08_soft_surfer.ipynb
