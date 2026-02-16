================
Soft Planktonics
================

.. image:: https://github.com/celoy/SoftMobility/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/celoy/SoftMobility/actions/workflows/testing.yml


.. image:: https://img.shields.io/pypi/v/SoftMobility.svg
        :target: https://pypi.python.org/pypi/SoftMobility


Python package to compute the mobility of soft plankton in Stokes flows.

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://celoy.github.io/SoftMobility.

Workflow
--------
Once the repository is cloned, it is advisable to run a virtualenv, with Python 3.8.18 or above.

Then to install the package `softmobility` properly, run

`pip install -e .`

To install the packages required for development, the command is

`pip install --upgrade -r requirements-dev.txt`

Features
--------

Developpers
-----------

Version number is changed with `git tag vX.Y.Z`


TODOS
-----
Documentation:

Development:
- TODO. Solve steady NL problem Mdof(theta)=0 
- TODO. Time dependent parameters or forces
- TODO. Adding constraints on DOFs (intervals)
- TODO. table lookout for soft motility problem
- TODO. Interactions between assemblies
- TODO. Assemblies of non-spherical particles
- TODO. General formulation of forces
- TODO. Units and properties (mu in flow property)
- TODO. Check problems arising when one sphere completely inside another

Unit tests:
- TODO. Unit test for flow and fluidplankton

Tutorials:
- TODO. Tutorial on surfing particles
- TODO. Tutorial on using automatic jax differentiation to optimize (isocahedron?)
- TODO. Tutorial on motility problems
- TODO. Tutorial on free fall 
- TODO. Tutorial on Jeffery orbits
- TODO. Example of time-stepping for a falling object

Research questions:
- TODO. test surf in the limit of Vswim<<1 with Lagrangian trajectories of passive particles
