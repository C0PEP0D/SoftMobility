__all__ = []

from .classes.sphere import Sphere
from .classes.sphereassembly import SphereAssembly
from .classes.softbody import SoftBody
from .classes.flow import Flow
from .classes.field import Field
from .classes.scalar import Scalar
from .classes.flowbodysolver import FlowBodySolver

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
