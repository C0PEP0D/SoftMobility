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

The package is installed as ``soft-mobility`` and imported as
``softmobility``.

Try the tutorials online
------------------------

The fastest way to get a feel for SoftMobility is to run a tutorial directly
in your browser via Google Colab — no clone, fork, or local install required.
The first cell of every tutorial installs ``SoftMobility`` from this
repository when it detects a Colab runtime (locally the cell is a no-op).

Click a badge below to launch the corresponding notebook in Colab:

* |colab-01| ``01_assembly_creation``
* |colab-02| ``02_rigid_mobility``
* |colab-03| ``03_soft_mobility_simulation``
* |colab-04| ``04_optimization``
* |colab-11| ``11_sinking_rigid_body``
* |colab-12| ``12_flexible_fiber_2d``
* |colab-13| ``13_rotating_fiber_3d``
* |colab-14| ``14_jeffery_rigid``
* |colab-21| ``21_jeffery_soft``
* |colab-22| ``22_three_sphere_swimmer``
* |colab-23| ``23_soft_surfer``

For any notebook not listed above, you can build a Colab URL by hand by
replacing the GitHub URL prefix
``https://github.com/C0PEP0D/SoftMobility/blob/`` with
``https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/``.

Installation
------------

SoftMobility requires Python 3.10 or newer. For a new user, the safest path is
to work in an isolated environment. Two equivalent recipes follow — pick the
one that matches the tooling you already use. The Sphinx ``installation``
page in the documentation contains the same recipes plus troubleshooting
notes (Apple Silicon, GPU JAX builds, etc.).

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

Tutorials
---------

Tutorial notebooks live in ``softmobility/tutorials`` and are grouped into
three layers; the numbering reflects the layer
(0X = library introduction, 1X = validation against published results,
2X = original case studies).

**Library introduction (0X)**

* ``01_assembly_creation.ipynb`` — methods to create a sphere assembly
* ``02_rigid_mobility.ipynb`` — mobility properties of a rigid sphere assembly
* ``03_soft_mobility_simulation.ipynb`` — soft mobility tensors and
  simulation of a trajectory
* ``04_optimization.ipynb`` — optimization principles

**Validation cases (1X)**

* ``11_sinking_rigid_body.ipynb`` — sinking trajectory of a rigid body
  *(work in progress)*
* ``12_flexible_fiber_2d.ipynb`` — 2-D flexible fiber in shear and gravity
  (Delmotte et al. 2015)
* ``13_rotating_fiber_3d.ipynb`` — 3-D filament: bending and rotational
  relaxation (Coq et al. 2008; Wiggins et al. 1998)
* ``14_jeffery_rigid.ipynb`` — Jeffery orbits of a rigid body

**Original case studies (2X)**

* ``21_jeffery_soft.ipynb`` — Jeffery orbit of a one-DOF deformable body
* ``22_three_sphere_swimmer.ipynb`` — three-sphere swimmer optimization
* ``23_soft_surfer.ipynb`` — soft surfer in Taylor–Green vortices

New users should start with ``01_assembly_creation`` and work through the
0X group before moving to validation cases or original studies.

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

.. |colab-01| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/01_assembly_creation.ipynb
.. |colab-02| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/02_rigid_mobility.ipynb
.. |colab-03| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/03_soft_mobility_simulation.ipynb
.. |colab-04| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/04_optimization.ipynb
.. |colab-11| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/11_sinking_rigid_body.ipynb
.. |colab-12| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/12_flexible_fiber_2d.ipynb
.. |colab-13| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/13_rotating_fiber_3d.ipynb
.. |colab-14| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/14_jeffery_rigid.ipynb
.. |colab-21| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/21_jeffery_soft.ipynb
.. |colab-22| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/22_three_sphere_swimmer.ipynb
.. |colab-23| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/C0PEP0D/SoftMobility/blob/main/softmobility/tutorials/23_soft_surfer.ipynb
