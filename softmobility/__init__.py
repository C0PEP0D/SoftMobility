__all__ = []

from .classes.sphere import Sphere
from .classes.sphereassembly import SphereAssembly
from .classes.softbody import SoftBody
from .classes.inputs import (
    Field,  # base classes
    Scalar,
    Flow,
    gravity_field,  # named field constructors
    rotating_magnetic_field,
    oscillating_magnetic_field,
    constant_scalar,  # named scalar constructors
    oscillating_scalar,
    no_flow,  # names flow constructors
    shear_flow,
    rotating_flow,
    extensional_flow,
)
from .classes.flowbodysolver import FlowBodySolver

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
