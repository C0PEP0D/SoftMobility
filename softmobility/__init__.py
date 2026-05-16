__all__ = []

from .classes.sphere import Sphere
from .classes.sphereassembly import SphereAssembly
from .classes.softbody import SoftBody
from .classes.flexiblefiber import FlexibleFiber
from .classes.extensiblefiber import ExtensibleFiber
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
    taylor_green_flow,
)
from .classes.solver import (
    FlowBodyRollout,
    FlowBodyOptimizer,
    FlowBodyRL,
    rotation_matrix,  # rotation utilities
    rescale_orientation,
    compute_bortz_operator,
)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
