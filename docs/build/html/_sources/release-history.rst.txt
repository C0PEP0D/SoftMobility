===============
Release History
===============

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
- ``PRFluids/`` directory renamed to ``manuscript/`` in tracked references.

v1.0.0 (2026-03-30)
-------------------

Initial public release of SoftMobility: a JAX-based library to simulate and
optimize the motion and deformation of soft bodies (assemblies of beads
connected by springs) in Stokes flow, with end-to-end differentiability for
gradient-based design.
