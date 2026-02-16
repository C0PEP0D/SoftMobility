import numpy as np
import yaml
import jax
import jax.numpy as jnp
import re
import sympy as sp
from io import StringIO
import os

from .sphere import Sphere


class SphereAssembly:
    def __init__(self, parameters_source: str = None, verbose=True):
        """
        Initialize SphereAssembly. Can be initialized from a parameters file or left empty.

        Parameters
        ----------
        parameters_source : str, optional
            Path to the YAML file containing parameters.
        verbose : bool, default=True
            Whether to print debug information.
        """
        # Initialize empty lists, allowing incremental sphere addition
        self._Ndof = 0  # integer
        self._Nparam = 0  # integer
        self._Nspheres = 0  # integer
        self._dof_variables = []  # list of str
        self._param_variables = []  # list of str
        self._dof_defaults = jnp.array([])
        self._param_defaults = jnp.array([])
        self.spheres = []  # List of Sphere objects

        # Load parameters if a file is provided
        if isinstance(parameters_source, str):
            self._initialize_from_file(parameters_source, verbose)

    def _initialize_from_file(self, parameters_source: str, verbose=True):
        """Helper method to parse the parameter file and initialize attributes."""
        (
            self._Ndof,
            self._Nparam,
            self._Nspheres,
            self._dof_variables,
            self._param_variables,
            self._dof_defaults,
            self._param_defaults,
            self.spheres,
        ) = self._parse_parameter_file(parameters_source, verbose)

        # check that forces and torques sum to zero
        sum_force = jnp.zeros((3,))
        sum_torque = jnp.zeros((3,))
        for sphere in self.spheres:
            sum_force += sphere.force(self.dof_defaults, self.param_defaults)
            sum_torque += sphere.torque(self.dof_defaults, self.param_defaults)

        if not jnp.allclose(sum_force, jnp.zeros((3,))):
            raise ValueError("Forces do not sum to zero.")

        if not jnp.allclose(sum_torque, jnp.zeros((3,))):
            raise ValueError("Torques do not sum to zero.")

    @classmethod
    def from_file(cls, parameters_source: str, verbose=True):
        """Alternative constructor to initialize from a YAML file."""
        return cls(parameters_source, verbose)

    def add_sphere(self, sphere: Sphere):
        """Add a Sphere object to the assembly."""
        # Check that the sphere has the correct number of degrees of freedom and parameters
        try:
            test_dof = jnp.zeros(self.Ndof)
            test_param = jnp.zeros(self.Nparam)

            sphere.radius(dofs=test_dof, params=test_param)
            sphere.position(dofs=test_dof, params=test_param)
            sphere.orientation(dofs=test_dof, params=test_param)
            sphere.force(dofs=test_dof, params=test_param)
            sphere.torque(dofs=test_dof, params=test_param)

        except ValueError as e:
            raise ValueError(f"Sphere does not have the correct number of degrees of freedom or parameters: {e}")

        self.spheres.append(sphere)
        self._Nspheres += 1  # Update sphere count

    def add_dof(self, name: str, default: float = 0.0):
        """Add a new degree of freedom (DOF)."""
        if name in self._dof_variables:
            raise ValueError(f"DOF '{name}' already exists.")

        self._dof_variables.append(name)
        self._Ndof += 1
        self._dof_defaults = jnp.append(self._dof_defaults, default)
        print(f"NEW degrees of freedom\n {self.dof_variables} \nwith default values\n {self.dof_defaults}")

    def add_param(self, name: str, default: float = 0.0):
        """Add a new parameter."""
        if name in self._param_variables:
            raise ValueError(f"Parameter '{name}' already exists.")

        self._param_variables.append(name)
        self._Nparam += 1
        self._param_defaults = jnp.append(self._param_defaults, default)
        print(f"NEW parameters\n {self.param_variables} \nwith default values\n {self.param_defaults}")

    # Read-only properties for fundamental attributes
    @property
    def Ndof(self):
        return self._Ndof

    @property
    def Nparam(self):
        return self._Nparam

    @property
    def Nspheres(self):
        return self._Nspheres

    @property
    def dof_variables(self):
        return self._dof_variables

    @property
    def param_variables(self):
        return self._param_variables

    @property
    def dof_defaults(self):
        return self._dof_defaults

    @property
    def param_defaults(self):
        return self._param_defaults

    def compute_stiffness_matrix(self, dofs=None, params=None):
        dofs, params = self._setup_params(dofs, params)
        stiffness_matrix = jax.jacfwd(self.grand_forces_func, argnums=0)
        K = stiffness_matrix(dofs, params)

        return K

    def compute_Jass(self, dofs=None, params=None):
        """
        Computes Jass, which is defined by V = Jass . dotQ, with V the grand velocity in the body's frame and dotQ the time derivative of the dofs.
        We use: V = B . dotX = B . J_X . dotQ, suh that Jass = B . J_X

        Args:
            dofs (np.array, optional): Degrees of freedom. Defaults to None.
            params (np.array, optional): Parameters. Defaults to None.

        Returns:
            jnp_array: Jass
        """
        dofs, params = self._setup_params(dofs, params)

        # Compute the Jacobian of X with respect to dofs Q using JAX's automatic differentiation
        Jacobian_X = jax.jacfwd(self.grand_coordinates_func, argnums=0)
        J_X = jnp.array(Jacobian_X(dofs, params))

        # Create the block-diagonal matrix N using a list
        # Each block is computed by bortz_jacobian_for_sphere for each sphere
        Bi_s = [sphere.bortz_jacobian(dofs, params) for sphere in self.spheres]

        # Construct N by stacking the blocks along the diagonal
        B = jnp.block(
            [
                [block if i == j else jnp.zeros_like(block) for j, block in enumerate(Bi_s)]
                for i in range(self.Nspheres)
            ]
        )

        # The final velocity matrix V is the product of N and M
        Jass = B @ J_X

        return Jass

    def compute_C_U(self, dofs=None, params=None):
        """
        Computes C_U, such that v = V + C_U .v_0, with V and v the grand velocity in the body and lab frame, and v_0 the six-component velocity of the body reference in the lab frame

        Args:
            dofs (np.array, optional): Degrees of freedom. Defaults to None.
            params (np.array, optional): Parameters. Defaults to None.

        Returns:
            jnp.array: C_U
        """
        dofs, params = self._setup_params(dofs, params)

        # Construct T by assembling vertically the individual T's for each sphere
        C_U = jnp.block([[sphere.composition_of_velocity(dofs, params)] for sphere in self.spheres])

        return C_U

    # def compute_composition_of_forces(self, dofs=None, params=None):
    #     dofs, params = self._setup_params(dofs, params)

    #     # Create blocks for individual spheres
    #     blocks = [sphere.composition_of_force(dofs, params) for sphere in self.spheres]

    #     # Construct Tf by stacking blocks along the diagonal
    #     Tf = jnp.block(
    #         [[b if i == j else jnp.zeros_like(b) for j, b in enumerate(blocks)] for i in range(self.Nspheres)]
    #     )

    #     return Tf

    def compute_Jacobian_matrix(self, dofs=None, params=None):
        """
        Computes the Jacobian tensor J = partial v / partial p

        Args:
            dofs (np.array, optional): Degrees of freedom. Defaults to None.
            params (np.array, optional): Parameters. Defaults to None.

        Returns:
            jnp.array: J
        """
        dofs, params = self._setup_params(dofs, params)

        Jass = self.compute_Jass(dofs, params)
        C_U = self.compute_C_U(dofs, params)
        J = jnp.block([C_U, Jass])

        return J

    def set_dof_defaults(self, new_dofs=None, new_dict=None, verbose=True):
        if verbose:
            print("OLD default dof values:", self.dof_variables, self.dof_defaults)
        if new_dofs is not None:
            try:
                new_dofs_array = jnp.array(new_dofs).flatten().astype(float)
            except TypeError as e:
                raise ValueError("Cannot cast new_dofs into a jnp array") from e

            if new_dofs_array.shape != (self.Ndof,):
                raise ValueError(f"new_dofs array must have shape ({self.Ndof},)")

            self._dof_defaults = new_dofs_array

        elif new_dict is not None:
            for key, value in new_dict.items():
                # Handle case where key doesn't exist in dof_variables
                if key not in self.dof_variables:
                    raise ValueError(f"Invalid variable name: {key}")

                # Ensure the value can be cast to float
                try:
                    idx = self.dof_variables.index(key)
                    new_value = float(value)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid value for variable '{key}': {value}")

                # Update the corresponding index in dof_defaults
                self._dof_defaults = self._dof_defaults.at[idx].set(new_value)
        if verbose:
            print("NEW default dof values:", self.dof_variables, self.dof_defaults)

    def set_param_defaults(self, new_params=None, new_dict=None, verbose=True):
        if verbose:
            print("OLD default param values:", self.param_variables, self.param_defaults)
        if new_params is not None:
            try:
                new_params_array = jnp.array(new_params).flatten().astype(float)
            except TypeError as e:
                raise ValueError("Cannot cast new_params into a jnp array") from e

            if new_params_array.shape != (self.Nparam,):
                raise ValueError(f"new_params array must have shape ({self.Nparam},)")

            self._param_defaults = new_params_array

        elif new_dict is not None:
            for key, value in new_dict.items():
                # Handle case where key doesn't exist in param_variables
                if key not in self.param_variables:
                    raise ValueError(f"Invalid variable name: {key}")

                # Ensure the value can be cast to float
                try:
                    idx = self.param_variables.index(key)
                    new_value = float(value)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid value for variable '{key}': {value}")

                # Update the corresponding index in param_defaults
                self._param_defaults = self._param_defaults.at[idx].set(new_value)
        if verbose:
            print("NEW default param values:", self.param_variables, self.param_defaults)

    def grand_radius_func(self, dofs=None, params=None):
        """Calculates the radius of each sphere based on input parameters."""
        dofs, params = self._setup_params(dofs, params)

        return jnp.concatenate([jnp.array([sphere.radius(dofs, params)]) for sphere in self.spheres])

    def grand_coordinates_func(self, dofs=None, params=None):
        """Computes X, the grand coordinate for all spheres simultaneously."""
        dofs, params = self._setup_params(dofs, params)
        coords = [
            jnp.concatenate([sphere.position(dofs, params), sphere.orientation(dofs, params)])
            for sphere in self.spheres
        ]
        return jnp.concatenate(coords)

    def grand_forces_func(self, dofs=None, params=None):
        """Computes f, the grand force for all spheres simultaneously."""
        dofs, params = self._setup_params(dofs, params)
        coords = [
            jnp.concatenate([sphere.force(dofs, params), sphere.torque(dofs, params)]) for sphere in self.spheres
        ]
        return jnp.concatenate(coords)

    def __str__(self):
        return f"Assembly with {self.Nspheres} spheres, {self.Ndof} degrees of freedom, and {self.Nparam} fixed parameters"

    def __repr__(self):
        """Print default values and details of sphere assembly."""
        output = "SPHERE ASSEMBLY\n"
        output += f"  {self.Nspheres} spheres\n"
        output += f"  {self.Ndof} degrees of freedom\n"
        output += f"  {self.Nparam} fixed parameters\n"

        # Print default values
        output += "\nDefault values\n"
        output += f"  degrees of freedom dof: {self.dof_variables} = {self.dof_defaults}\n"
        output += f"  fixed parameters param: {self.param_variables} = {self.param_defaults}\n"

        # Print example sphere assembly
        for i, sphere in enumerate(self.spheres):
            output += f"\nSPHERE {i}\n"
            output += f"  radius: {sphere.radius(self.dof_defaults, self.param_defaults)}\n"
            output += f"  position: {sphere.position(self.dof_defaults, self.param_defaults)}\n"
            output += f"  orientation: {sphere.orientation(self.dof_defaults, self.param_defaults)}\n"
            output += f"  force: {sphere.force(self.dof_defaults, self.param_defaults)}\n"
            output += f"  torque: {sphere.torque(self.dof_defaults, self.param_defaults)}\n"

        return output

    def _setup_params(self, dofs=None, params=None):
        """Private helper to setup default parameters"""
        if dofs is None:
            dofs = self.dof_defaults
        if params is None:
            params = self.param_defaults
        return jnp.array(dofs).astype(float), jnp.array(params).astype(float)

    def _parse_parameter_file(self, parameters_source: str, verbose: bool):
        """
        Parse the YAML parameter file to generate the coordinates function and radii of a sphere assembly.

        Parameters
        ----------
        parameters_source : str
            Path to the YAML file (if is_file=True) or a YAML string (if is_file=False).
        verbose : bool, default=True
            Whether to print debug information.

        Returns
        -------
        Ndof : int
            Number of degrees of freedom.
        Nparam : int
            Number of parameters.
        Nspheres : int
            Number of spheres in the assembly.
        dof_variables : list[str]
            List of dof variable names.
        param_variables : list[str]
            List of parameter variable names.
        dof_defaults : jax.numpy.ndarray
            Default array of values for dof.
        param_defaults : jax.numpy.ndarray
            Default array of values for param.
        sphere : list[Sphere]
            List of Sphere objects representing each sphere in the assembly.
        """
        # Read YAML data from file or string
        try:
            # Check if the input is a file path (ends with .yaml or .yml, case insensitive)
            if parameters_source.lower().endswith((".yaml", ".yml")):
                if not os.path.exists(parameters_source):
                    raise FileNotFoundError(f"File not found: {parameters_source}")
                with open(parameters_source, "r") as f:
                    yaml_data = yaml.safe_load(f)
                    if verbose:
                        print(f"Parsing parameter file: {parameters_source}")
            else:
                yaml_data = yaml.safe_load(StringIO(parameters_source))

            if not isinstance(yaml_data, dict):
                raise ValueError("Invalid YAML format: Expected a dictionary at the root level.")

        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            raise ValueError(f"Error loading parameters: {e}")

        # Extract sphere data
        sphere_data = yaml_data["spheres"]

        # Extract all possible prefixes from dof_names and param_names
        dof_prefixes = set(["dof"])
        param_prefixes = set(["param"])

        if "dof_names" in yaml_data:
            for name in yaml_data["dof_names"]:
                dof_prefixes.add(name)

        if "param_names" in yaml_data:
            for name in yaml_data["param_names"]:
                param_prefixes.add(name)

        # Extract variables from expressions
        dof_set = set()
        param_set = set()

        # In the section where you process sphere_data:
        for data in sphere_data:
            # Convert all components to strings and give default values if absent
            radius = [str(data.get("radius", 1))]
            position = [str(x) for x in data.get("position", [0, 0, 0])]
            orientation = [str(x) for x in data.get("orientation", [0, 0, 0])]
            force = [str(x) for x in data.get("force", [0, 0, 0])]
            torque = [str(x) for x in data.get("torque", [0, 0, 0])]

            # Use these converted lists instead of raw data
            sphere_exprs = radius + position + orientation + force + torque

            for expr in sphere_exprs:
                for dof_prefix in dof_prefixes:
                    dof_set.update(re.findall(r"(?:" + dof_prefix + ")(?:\d+)?", expr))
                for param_prefix in param_prefixes:
                    param_set.update(re.findall(r"(?:" + param_prefix + ")(?:\d+)?", expr))

        # Transform variable sets into lists and sort them alphabetically
        dof_variables = sorted(list(dof_set))
        param_variables = sorted(list(param_set))

        # printing variables found
        if verbose:
            print(
                f"  Found variables: {', '.join(dof_variables) + ', ' if dof_variables else ''}{', '.join(param_variables)}"
            )

        # Determine Ndof, Nparam, and Nsphere
        Ndof = len(dof_variables)
        Nparam = len(param_variables)
        Nsphere = len(sphere_data)

        # Create default arrays with zeros
        dof_defaults = jnp.zeros(Ndof)
        param_defaults = jnp.zeros(Nparam)

        # Read default values from parameters.yaml
        if "defaults" in yaml_data:
            defaults = yaml_data.get("defaults", {})

            # Replace default values in arrays
            for i, var in enumerate(dof_variables):
                dof_defaults = dof_defaults.at[i].set(defaults.get(var, 0.0))

            for i, var in enumerate(param_variables):
                param_defaults = param_defaults.at[i].set(defaults.get(var, 0.0))

        # Generate functions for generalized coordinates for each sphere
        spheres = []

        for i, data in enumerate(sphere_data):
            # Handle missing entries and str all
            rad_exprs = str(data.get("radius", 1))
            pos_exprs = [str(x) for x in data.get("position", [0, 0, 0])]
            ori_exprs = [str(x) for x in data.get("orientation", [0, 0, 0])]
            for_exprs = [str(x) for x in data.get("force", [0, 0, 0])]
            tor_exprs = [str(x) for x in data.get("torque", [0, 0, 0])]

            # Create a function for each sphere
            rad_func = create_function(rad_exprs, dof_variables, param_variables)
            pos_func = create_function(pos_exprs, dof_variables, param_variables)
            ori_func = create_function(ori_exprs, dof_variables, param_variables)
            for_func = create_function(for_exprs, dof_variables, param_variables)
            tor_func = create_function(tor_exprs, dof_variables, param_variables)
            spheres.append(Sphere(rad_func, pos_func, ori_func, for_func, tor_func))

            # Printing the characteristics of each sphere
            if verbose:
                print(f"    Sphere {i}")
                print(f"      Radius: {rad_func.expression}")
                print(f"      Position: {pos_func.expression}")
                print(f"      Orientation: {ori_func.expression}")
                print(f"      Force: {for_func.expression}")
                print(f"      Torque: {tor_func.expression}")

        return (
            Ndof,
            Nparam,
            Nsphere,
            dof_variables,
            param_variables,
            dof_defaults,
            param_defaults,
            spheres,
        )


# Useful functions ###########################################################


def create_function(sp_exprs, dofs, params):
    """
    Create a function that takes dofs and yaml_data as input and returns the evaluated symbolic expressions sp_exprs.

    Parameters
    ----------
    sp_exprs : str or list[str]
        A sympy expression or a list of sympy expressions.
    dofs : list[sympy.Symbol]
        A list of symbolic variable names for dofs.
    yaml_data : list[sympy.Symbol]
        A list of symbolic variable names for yaml_data.

    Returns
    -------
    callable
        A function that takes dofs and yaml_data as input and returns the evaluated expression.
    """
    dof_symbols = [sp.symbols(t) for t in dofs]
    param_symbols = [sp.symbols(p) for p in params]

    if isinstance(sp_exprs, str):
        # Parse the expression using sympy
        sp_expr = sp.sympify(sp_exprs)

        # Convert expression to a JAX function or raise ValueError
        try:
            jax_expr = sp.lambdify([dof_symbols, param_symbols], sp_expr, "jax")
        except Exception as e:
            raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}")

        # Callable build out of the JAX function
        def wrapper(dof_args, param_args):
            result = jax_expr(dof_args, param_args)
            if not isinstance(jax_expr(dof_args, param_args), (float)):
                result = jnp.array(result, float)
                result = result[(0,) * result.ndim]
            return result

        wrapper.expression = str(sp_expr)  # str(jax_expr(*dofs, *yaml_data))
        return wrapper
    else:
        # Parse the expressions using sympy
        sp_exprs = [sp.sympify(expr) for expr in sp_exprs]
        jax_exprs = []

        # Convert each expression to a JAX function and append it to the list of functions
        for sp_expr in sp_exprs:
            try:
                jax_exprs.append(sp.lambdify([dof_symbols, param_symbols], sp_expr, "jax"))
            except Exception as e:
                raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}")

        def wrapper(dof_args, param_args):
            list_expr = [
                (
                    jax_expr(dof_args, param_args).item()
                    if isinstance(jax_expr(dof_args, param_args), (list, np.ndarray))
                    else jax_expr(dof_args, param_args)
                )
                for jax_expr in jax_exprs
            ]
            return jnp.array(list_expr, dtype=float)

        wrapper.expression = [str(sp_expr) for sp_expr in sp_exprs]
        return wrapper
