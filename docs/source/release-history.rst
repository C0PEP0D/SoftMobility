===============
Release History
===============

v1.2.0 (2026-09-17)
-------------------

New features
~~~~~~~~~~~~

- Implicit time-integration schemes in ``FlowBodyRollout.rollout``:
  ``scheme="implicit_midpoint"`` (A-stable, ``O(dt^2)``, non-dissipative) and
  ``scheme="backward_euler"`` (L-stable, ``O(dt)``). Both remain stable for
  stiff internal dynamics — very stiff springs — at time steps far beyond the
  explicit stability limit ``dt < O(1/k)``, where the Runge-Kutta schemes
  diverge.
- New ``n_newton`` argument (default 3) controlling the number of Newton
  iterations per implicit step. The count is static and the Jacobian is a
  dense ``(6 + Ndof)`` forward-mode AD evaluation, so the step stays
  ``jax.jit``, ``lax.scan``, and ``vmap`` compatible.

Improvements
~~~~~~~~~~~~

- Notation in examples 01 and 07 aligned with the revised companion paper.
- New GPU acceleration section for local installs.
- ``pip install softmobility`` documented as the primary installation route;
  Colab cells now install from PyPI and request a GPU runtime.
- Citation section with BibTeX entries added to the README.

Fixes
~~~~~

- Duplicate JAX-guide target that broke the docs build under stricter
  docutils.
- Stale references to the in-tree manuscript removed from the documentation.

Internal
~~~~~~~~

- ``nbstripout`` enabled as a pre-commit hook; notebook outputs stripped and
  notebook source format normalised across tutorials and examples.
- Developer-local files, build artifacts, ``docs/build/``, and ``manuscript/``
  untracked.
- GitHub Actions runners bumped (``checkout@v5``, ``setup-python@v6``).
- Test coverage added for the rollout integration schemes in
  ``softmobility/tests/test_class_flowbodyrollout.py``.
- Machine-readable citation metadata added: ``CITATION.cff`` and
  ``.zenodo.json``. Releases from this one on are archived on Zenodo with a
  DOI.

v1.1.0 (2026-05-21)
-------------------

New features
~~~~~~~~~~~~

- Intrinsic curvature support for soft bodies.
- Four-sphere magnetic swimmer example and supporting machinery.
- Time-dependent forcing in rollouts.
- Differentiable Bortz operators (gradients now flow through).
- Rigid-body mobility tensors exposed as a first-class utility.
- Opt-in GRPY overlap regimes via the ``allow_overlap`` flag.
- Flexible fibers: clamped boundary condition, settling fiber, gears model.
- ``rotation_matrix`` exposed at the package level.

Improvements
~~~~~~~~~~~~

- Plotting migrated from plotly to matplotlib across tutorials, examples,
  and the public API.
- New ``softmobility.classes.figstyle`` module providing the paper aesthetic
  (``figstyle.apply()`` in notebooks).
- Tutorials reorganised: pedagogical notebooks under ``tutorials/`` and
  validation/case studies under ``examples/``.
- New validation case studies and a Results-section validation prose draft.
- README restructured for new users; new ``Developers`` documentation page
  covering build, docs, versioning, and release workflow.
- Conda and Google Colab installation paths documented.
- Docstrings added across ``SoftBody`` and the solver module.
- ``optimize`` now supports a no-overlap option and ``vmap`` batching;
  ``add_sphere`` corrected.

Fixes
~~~~~

- Clamped flexible fiber simulation bug.
- Google Colab file-import bug in example notebooks.
- ``figstyle.save`` now gracefully skips PDF export when ``kaleido`` is
  missing.
- Docs build cleanups: orphan autosummary stubs removed, ``index.rst`` typo
  fixed, regenerated figure PDFs untracked.
- Repository URLs updated from ``celoy/SoftMobility`` to
  ``C0PEP0D/SoftMobility``.

Internal
~~~~~~~~

- Ruff cleanup across the codebase and notebooks.
- ``tox`` configuration removed.

v1.0.0 (2026-03-30)
-------------------

Initial public release of SoftMobility: a JAX-based library to simulate and
optimize the motion and deformation of soft bodies (assemblies of beads
connected by springs) in Stokes flow, with end-to-end differentiability for
gradient-based design.
