========
Geometry
========

Geometry is described by ``Sphere`` objects collected in a
``SphereAssembly``. A sphere can be built directly from constants and callables,
or indirectly from a YAML description parsed by ``SphereAssembly`` and
``SoftBody``.

Spheres
-------

``Sphere`` stores five callable quantities:

``radius(dofs, design)``
    Radius of the sphere.

``position(dofs, design, time)``
    Body-frame position of the sphere center.

``orientation(dofs, design, time)``
    Sphere orientation as a Rodrigues vector.

``c_field(dofs, design)``
    Coupling matrix from external inputs to sphere force and torque.

``c_stiff(dofs, design)``
    Coupling matrix from degrees of freedom to elastic force and torque.

Constants, arrays, and callables are accepted where possible. Callables must be
JAX-compatible when they will be used in differentiated simulations.

YAML geometry
-------------

YAML files are the most compact way to describe reusable scientific models. The
parser detects symbols by prefixes:

.. code-block:: yaml

   dof_names:
     - x
   design_names:
     - radius
     - length
     - k
   input_names:
     - gravity

   defaults:
     x0: 0.1
     radius: 0.25
     length: 1.0
     k: 1.0

   spheres:
     - radius: radius
       position: [-length / 2, 0, 0]
       orientation: [0, x0, 0]
       force: [gravity0, gravity1, gravity2]
       torque: [0, -k * x0, 0]
     - radius: radius
       position: [length / 2, 0, 0]
       orientation: [0, -x0, 0]
       force: [-gravity0, -gravity1, -gravity2]
       torque: [0, k * x0, 0]

The same string can be passed directly to ``SphereAssembly`` or ``SoftBody``:

.. code-block:: python

   from softmobility import SoftBody

   body = SoftBody(yaml_text, verbose=False)

The parser canonicalizes variables alphabetically within each group. Inspect
``body.dof_variables``, ``body.design_variables``, and
``body.input_variables`` before passing arrays explicitly.

Assembly matrices
-----------------

``SphereAssembly`` provides low-level matrices used by ``SoftBody``:

``compute_Jassembly``
    Maps degree-of-freedom rates to sphere velocities in the body frame.

``compute_C_U``
    Maps body reference velocity to the grand velocity of all spheres.

``compute_Jacobian_matrix``
    Builds the complete kinematic Jacobian used by the reduced mobility
    problem.

Most users call these methods for diagnostics or method development. For
simulation, use ``SoftBody.compute_tensors`` through ``FlowBodyRollout``.
