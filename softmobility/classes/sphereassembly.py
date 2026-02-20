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
        self._Ndesign = 0  # integer
        self._Ninput = 0  # integer
        self._Nspheres = 0  # integer
        self._dof_variables = []  # list of str
        self._design_variables = []  # list of str
        self._input_variables = []  # list of str
        self._dof_defaults = jnp.array([])
        self._design_defaults = jnp.array([])
        self._input_defaults = jnp.array([])
        self.spheres = []  # List of Sphere objects

        # Load parameters if a file is provided
        if isinstance(parameters_source, str):
            self._initialize_from_file(parameters_source, verbose)

    def _initialize_from_file(self, parameters_source: str, verbose=True):
        """Helper method to parse the parameter file and initialize attributes."""
        (
            self._Ndof,
            self._Ndesign,
            self._Ninput,
            self._Nspheres,
            self._dof_variables,
            self._design_variables,
            self._input_variables,
            self._dof_defaults,
            self._design_defaults,
            self.spheres,
        ) = self._parse_parameter_file(parameters_source, verbose)

    @classmethod
    def from_file(cls, parameters_source: str, verbose=True):
        """Alternative constructor to initialize from a YAML file."""
        return cls(parameters_source, verbose)

    def add_sphere(self, sphere: Sphere):
        """Add a Sphere object to the assembly."""
        # Check that the sphere has the correct number of degrees of freedom and parameters
        try:
            test_dof = jnp.zeros(self.Ndof)
            test_design = jnp.zeros(self.Ndesign)

            sphere.radius(dofs=test_dof, design=test_design)
            sphere.position(dofs=test_dof, design=test_design)
            sphere.orientation(dofs=test_dof, design=test_design)
            sphere.c_field(dofs=test_dof, design=test_design)
            sphere.c_stiff(dofs=test_dof, design=test_design)

        except ValueError as e:
            raise ValueError(f"Sphere does not have the correct number of degrees of freedom or variables: {e}")

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

    def add_design(self, name: str, default: float = 0.0):
        """Add a new design parameter."""
        if name in self._design_variables:
            raise ValueError(f"Design parameter '{name}' already exists.")

        self._design_variables.append(name)
        self._Ndesign += 1
        self._design_defaults = jnp.append(self._design_defaults, default)
        print(f"NEW design parameters\n {self.design_variables} \nwith default values\n {self.design_defaults}")

    # Read-only properties for fundamental attributes
    @property
    def Ndof(self):
        return self._Ndof

    @property
    def Ndesign(self):
        return self._Ndesign

    @property
    def Ninput(self):
        return self._Ninput

    @property
    def Nspheres(self):
        return self._Nspheres

    @property
    def dof_variables(self):
        return self._dof_variables

    @property
    def design_variables(self):
        return self._design_variables

    @property
    def input_variables(self):
        return self._input_variables

    @property
    def dof_defaults(self):
        return self._dof_defaults

    @property
    def design_defaults(self):
        return self._design_defaults

    def grand_c_field(self, dofs=None, design=None):
        """
        Returns C_field of shape (6N, Ninput) such that:
            grand_force_torque = C_field @ inputs + C_stiff @ dofs
        """
        dofs, design = self._setup_params(dofs, design)
        return jnp.vstack([sphere.c_field(dofs, design) for sphere in self.spheres])

    def grand_c_stiff(self, dofs=None, design=None):
        """
        Returns C_stiff of shape (6N, Ndof) such that:
            grand_force_torque = C_field @ inputs + C_stiff @ dofs
        """
        dofs, design = self._setup_params(dofs, design)
        return jnp.vstack([sphere.c_stiff(dofs, design) for sphere in self.spheres])

    def compute_Jass(self, dofs=None, design=None):
        """
        Computes Jass, which is defined by V = Jass . dotQ, with V the grand velocity in the body's frame and dotQ the time derivative of the dofs.
        We use: V = B . dotX = B . J_X . dotQ, suh that Jass = B . J_X

        Args:
            dofs (jnp.ndarray, optional): Degrees of freedom. Defaults to None.
            design (jnp.ndarray, optional): Design Parameters. Defaults to None.

        Returns:
            jnp_array: Jass
        """
        dofs, design = self._setup_params(dofs, design)

        # Compute the Jacobian of X with respect to dofs Q using JAX's automatic differentiation
        Jacobian_X = jax.jacfwd(self.grand_coordinates_func, argnums=0)
        J_X = jnp.array(Jacobian_X(dofs, design))

        # Create the block-diagonal
        # Each block is computed by bortz_jacobian_for_sphere for each sphere
        Bi_s = [sphere.bortz_jacobian(dofs, design) for sphere in self.spheres]

        # Construct N by stacking the blocks along the diagonal
        B = jnp.block(
            [
                [block if i == j else jnp.zeros_like(block) for j, block in enumerate(Bi_s)]
                for i in range(self.Nspheres)
            ]
        )

        Jass = B @ J_X

        return Jass

    def compute_C_U(self, dofs=None, design=None):
        """
        Computes C_U, such that v = V + C_U .v_0, with V and v the grand velocity in the body and lab frame, and v_0 the six-component velocity of the body reference in the lab frame

        Args:
            dofs (jnp.ndarray, optional): Degrees of freedom. Defaults to None.
            design (jnp.ndarray, optional): Design Parameters. Defaults to None.

        Returns:
            jnp.array: C_U
        """
        dofs, design = self._setup_params(dofs, design)

        # Construct T by assembling vertically the individual T's for each sphere
        C_U = jnp.block([[sphere.composition_of_velocity(dofs, design)] for sphere in self.spheres])

        return C_U

    def compute_composition_of_forces(self, dofs=None, design=None):
        dofs, design = self._setup_params(dofs, design)

        # Create blocks for individual spheres
        blocks = [sphere.composition_of_force(dofs, design) for sphere in self.spheres]

        # Construct Tf by stacking blocks along the diagonal
        Tf = jnp.block(
            [[b if i == j else jnp.zeros_like(b) for j, b in enumerate(blocks)] for i in range(self.Nspheres)]
        )

        return Tf

    def compute_Jacobian_matrix(self, dofs=None, design=None):
        """
        Computes the Jacobian tensor J = partial v / partial p

        Args:
            dofs (jnp.ndarray, optional): Degrees of freedom. Defaults to None.
            design (jnp.ndarray, optional): Design Parameters. Defaults to None.

        Returns:
            jnp.array: J
        """
        dofs, design = self._setup_params(dofs, design)

        Jass = self.compute_Jass(dofs, design)
        C_U = self.compute_C_U(dofs, design)
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

    def set_design_defaults(self, new_design=None, new_dict=None, verbose=True):
        if verbose:
            print("OLD default param values:", self.design_variables, self.design_defaults)
        if new_design is not None:
            try:
                new_design_array = jnp.array(new_design).flatten().astype(float)
            except TypeError as e:
                raise ValueError("Cannot cast new_design into a jnp array") from e

            if new_design_array.shape != (self.Ndesign,):
                raise ValueError(f"new_design array must have shape ({self.Ndesign},)")

            self._design_defaults = new_design_array

        elif new_dict is not None:
            for key, value in new_dict.items():
                # Handle case where key doesn't exist in design_variables
                if key not in self.design_variables:
                    raise ValueError(f"Invalid variable name: {key}")

                # Ensure the value can be cast to float
                try:
                    idx = self.design_variables.index(key)
                    new_value = float(value)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid value for variable '{key}': {value}")

                # Update the corresponding index in design_defaults
                self._design_defaults = self._design_defaults.at[idx].set(new_value)
        if verbose:
            print("NEW default param values:", self.design_variables, self.design_defaults)

    def grand_radius_func(self, dofs=None, design=None):
        """Calculates the radius of each sphere based on input parameters."""
        dofs, design = self._setup_params(dofs, design)

        return jnp.concatenate([jnp.array([sphere.radius(dofs, design)]) for sphere in self.spheres])

    def grand_coordinates_func(self, dofs=None, design=None):
        """Computes X, the grand coordinate for all spheres simultaneously."""
        dofs, design = self._setup_params(dofs, design)
        coords = [
            jnp.concatenate([sphere.position(dofs, design), sphere.orientation(dofs, design)])
            for sphere in self.spheres
        ]
        return jnp.concatenate(coords)

    def __str__(self):
        return f"Assembly with {self.Nspheres} spheres, {self.Ndof} degrees of freedom, and {self.Ndesign} fixed parameters"

    def __repr__(self):
        """Print default values and details of sphere assembly."""
        output = "SPHERE ASSEMBLY\n"
        output += f"  {self.Nspheres} spheres\n"
        output += f"  {self.Ndof} degrees of freedom\n"
        output += f"  {self.Ndesign} design parameters\n"
        output += f"  {self.Ninput} input parameters\n"

        # Print default values
        output += "\nDefault values\n"
        output += f"  degrees of freedom dof: {self.dof_variables} = {self.dof_defaults}\n"
        output += f"  design parameters param: {self.design_variables} = {self.design_defaults}\n"
        output += f"  input parameters param: {self.input_variables}\n"

        # Print example sphere assembly
        for i, sphere in enumerate(self.spheres):
            output += f"\nSPHERE {i}\n"
            output += f"  radius: {sphere.radius(self.dof_defaults, self.design_defaults)}\n"
            output += f"  position: {sphere.position(self.dof_defaults, self.design_defaults)}\n"
            output += f"  orientation: {sphere.orientation(self.dof_defaults, self.design_defaults)}\n"
            output += f"  c_field:\n{sphere.c_field(self.dof_defaults, self.design_defaults)}\n"
            output += f"  c_stiff:\n{sphere.c_stiff(self.dof_defaults, self.design_defaults)}\n"

        return output

    def _setup_params(self, dofs=None, design=None):
        """Private helper to setup default parameters"""
        if dofs is None:
            dofs = self.dof_defaults
        if design is None:
            design = self.design_defaults
        return jnp.array(dofs).astype(float), jnp.array(design).astype(float)

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
        Ndesign : int
            Number of design parameters.
        Ninput : int
            Number of input parameters.
        Nspheres : int
            Number of spheres in the assembly.
        dof_variables : list[str]
            List of dof variable names.
        design_variables : list[str]
            List of design parameter variable names.
        input_variables : list[str]
            List of input parameter variable names.
        dof_defaults : jax.numpy.ndarray
            Default array of values for dof.
        design_defaults : jax.numpy.ndarray
            Default array of values for design param.
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

        # Extract all possible prefixes from dof_names and design_names
        dof_prefixes = set(["dof"])
        design_prefixes = set(["design"])
        input_prefixes = set(["input"])

        if "dof_names" in yaml_data:
            for name in yaml_data["dof_names"]:
                dof_prefixes.add(name)

        if "design_names" in yaml_data:
            for name in yaml_data["design_names"]:
                design_prefixes.add(name)

        if "input_names" in yaml_data:
            for name in yaml_data["input_names"]:
                input_prefixes.add(name)

        # Extract variables from expressions
        dof_set = set()
        design_set = set()
        input_set = set()

        # In the section where you process sphere_data:
        all_detected_symbols = set()
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
                for design_prefix in design_prefixes:
                    design_set.update(re.findall(r"(?:" + design_prefix + ")(?:\d+)?", expr))
                for input_prefix in input_prefixes:
                    input_set.update(re.findall(r"(?:" + input_prefix + ")(?:\d+)?", expr))

            for expr in sphere_exprs:
                expr_symbols = {str(s) for s in sp.sympify(expr).free_symbols}
                all_detected_symbols.update(expr_symbols)

        # Transform variable sets into lists and sort them alphabetically
        dof_variables = sorted(list(dof_set))
        design_variables = sorted(list(design_set))
        input_variables = sorted(list(input_set))

        # Classify and validate input variable names
        field3d_bases, scalar_names = _classify_input_variables(input_variables)

        # Reorder canonically: field components first (in field order), scalars last
        input_variables = [base + str(j) for base in field3d_bases for j in range(3)] + scalar_names
        Ninput = len(input_variables)

        # printing variables found
        if verbose:
            print(
                f"  Found variables: {', '.join(dof_variables) + ', ' if dof_variables else ''}{', '.join(design_variables) + ', ' if design_variables else ''}{', '.join(input_variables)}"
            )
            if field3d_bases:
                print(f"  3D field inputs:  {field3d_bases}")
            if scalar_names:
                print(f"  Scalar inputs:    {scalar_names}")

        # Determine Ndof, Ndesign, and Nsphere
        Ndof = len(dof_variables)
        Ndesign = len(design_variables)
        Ninput = len(input_variables)
        Nsphere = len(sphere_data)

        # Create default arrays with zeros
        dof_defaults = jnp.zeros(Ndof)
        design_defaults = jnp.zeros(Ndesign)

        # Extract constants = defaults that are not dof/design/input
        constants = {}

        # Read default values from parameters.yaml
        if "defaults" in yaml_data:
            defaults = yaml_data.get("defaults", {})

            for key, value in defaults.items():
                if key not in dof_variables and key not in design_variables and key not in input_variables:
                    constants[key] = value

            # Replace default values in arrays
            for i, var in enumerate(dof_variables):
                dof_defaults = dof_defaults.at[i].set(defaults.get(var, 0.0))

            for i, var in enumerate(design_variables):
                design_defaults = design_defaults.at[i].set(defaults.get(var, 0.0))

        # Detect differences between declared symbols are used ones
        known_symbols = set(dof_variables) | set(design_variables) | set(input_variables) | set(constants.keys())

        undefined_symbols = all_detected_symbols - known_symbols

        if undefined_symbols:
            raise ValueError("Undefined symbols in expressions: " + ", ".join(sorted(undefined_symbols)))

        if verbose:
            unused_defaults = known_symbols - all_detected_symbols

            if unused_defaults:
                print("  Warning: defaults declared but not used: " + ", ".join(sorted(unused_defaults)))

        # Validate input variable usage
        _validate_inputs(sphere_data, input_variables, constants)

        # Generate functions for generalized coordinates for each sphere
        spheres = []

        # Create a function for each sphere
        for i, data in enumerate(sphere_data):
            rad_exprs = str(data.get("radius", 1))
            pos_exprs = [str(x) for x in data.get("position", [0, 0, 0])]
            ori_exprs = [str(x) for x in data.get("orientation", [0, 0, 0])]
            for_exprs = [str(x) for x in data.get("force", [0, 0, 0])]
            tor_exprs = [str(x) for x in data.get("torque", [0, 0, 0])]

            rad_func = create_function(rad_exprs, dof_variables, design_variables, constants)
            pos_func = create_function(pos_exprs, dof_variables, design_variables, constants)
            ori_func = create_function(ori_exprs, dof_variables, design_variables, constants)
            C_field_func, C_stiff_func = create_coupling_functions(
                for_exprs + tor_exprs, dof_variables, design_variables, input_variables, constants
            )
            spheres.append(Sphere(rad_func, pos_func, ori_func, C_field_func, C_stiff_func))

            # Printing the characteristics of each sphere
            if verbose:
                print(f"    Sphere {i}")
                print(f"      Radius: {rad_func.expression}")
                print(f"      Position: {pos_func.expression}")
                print(f"      Orientation: {ori_func.expression}")
                print(f"      Coupling matrix C_H:\n{C_field_func.expression}")
                print(f"      Coupling matrix C_K:\n{C_stiff_func.expression}")

        return (
            Ndof,
            Ndesign,
            Ninput,
            Nsphere,
            dof_variables,
            design_variables,
            input_variables,
            dof_defaults,
            design_defaults,
            spheres,
        )


# Useful functions ###########################################################


def create_function(sp_exprs, dofs, design, constants):
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
    design_symbols = [sp.symbols(p) for p in design]

    if isinstance(sp_exprs, str):
        # Parse the expression using sympy
        sp_expr = sp.sympify(sp_exprs)
        sp_expr = sp_expr.subs(constants)

        # Convert expression to a JAX function or raise ValueError
        try:
            jax_expr = sp.lambdify([dof_symbols, design_symbols], sp_expr, "jax")
        except Exception as e:
            raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}")

        # Callable build out of the JAX function
        def wrapper(dof_args, design_args):
            result = jax_expr(dof_args, design_args)
            if not isinstance(jax_expr(dof_args, design_args), (float)):
                result = jnp.array(result, float)
                result = result[(0,) * result.ndim]
            return result

        wrapper.expression = str(sp_expr)  # str(jax_expr(*dofs, *yaml_data))
        return wrapper
    else:
        # Parse the expressions using sympy
        sp_exprs = [sp.sympify(expr).subs(constants) for expr in sp_exprs]
        jax_exprs = []

        # Convert each expression to a JAX function and append it to the list of functions
        for sp_expr in sp_exprs:
            try:
                jax_exprs.append(sp.lambdify([dof_symbols, design_symbols], sp_expr, "jax"))
            except Exception as e:
                raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}")

        def wrapper(dof_args, design_args):
            list_expr = [
                (
                    jax_expr(dof_args, design_args).item()
                    if isinstance(jax_expr(dof_args, design_args), (list, np.ndarray))
                    else jax_expr(dof_args, design_args)
                )
                for jax_expr in jax_exprs
            ]
            return jnp.array(list_expr, dtype=float)

        wrapper.expression = [str(sp_expr) for sp_expr in sp_exprs]
        return wrapper


def create_coupling_functions(sp_exprs, dof_variables, design_variables, input_variables, constants):
    dof_symbols = [sp.Symbol(v) for v in dof_variables]
    design_symbols = [sp.Symbol(v) for v in design_variables]
    input_symbols = [sp.Symbol(v) for v in input_variables]

    exprs = [sp.sympify(e).subs(constants) for e in sp_exprs]

    # --- Validate linearity in inputs ---
    for i, expr in enumerate(exprs):
        for sym in input_symbols:
            if not sp.simplify(sp.diff(expr, sym, 2)).is_zero:
                raise ValueError(f"Expression {i} '{expr}' is nonlinear in input '{sym}'.")

    # --- C_field[i, j] = d(expr_i) / d(input_j) ---
    C_field_sym = [[sp.diff(expr, inp) for inp in input_symbols] for expr in exprs]

    # --- Subtract field contribution: new_expr = expr - C_field @ inputs ---
    residual_sym = [
        sp.simplify(expr - sum(c * inp for c, inp in zip(row, input_symbols)))
        for expr, row in zip(exprs, C_field_sym)
    ]

    # --- Validate that residual has no remaining input dependence ---
    for i, res in enumerate(residual_sym):
        for sym in input_symbols:
            if not sp.simplify(sp.diff(res, sym)).is_zero:
                raise ValueError(
                    f"Expression {i}: residual '{res}' still depends on input '{sym}' "
                    f"after subtracting C_field @ inputs. Check linearity."
                )

    # --- C_stiff[i, j] = d(residual_i) / d(dof_j) ---
    C_stiff_sym = [[sp.diff(res, dof) for dof in dof_symbols] for res in residual_sym]

    # --- Lambdify ---
    C_field_funcs = [[sp.lambdify([dof_symbols, design_symbols], c, "jax") for c in row] for row in C_field_sym]
    C_stiff_funcs = [[sp.lambdify([dof_symbols, design_symbols], c, "jax") for c in row] for row in C_stiff_sym]

    def C_field_func(dof_args, design_args):
        return jnp.array(
            [[c(dof_args, design_args) for c in row] for row in C_field_funcs], dtype=float
        )  # shape (6, Ninput)

    def C_stiff_func(dof_args, design_args):
        return jnp.array(
            [[c(dof_args, design_args) for c in row] for row in C_stiff_funcs], dtype=float
        )  # shape (6, Ndof)

    C_field_func.expression = [[str(c) for c in row] for row in C_field_sym]
    C_stiff_func.expression = [[str(c) for c in row] for row in C_stiff_sym]

    return C_field_func, C_stiff_func
    # d/d(dof)


def _classify_input_variables(input_variables: list[str]) -> tuple[list[str], list[str]]:
    """
    Classify input variables into 3D fields and scalars based on their suffixes.

    Rules:
    - A base name appearing with numeric suffix(es) → 3D field.
      All three components (base0, base1, base2) must be present, no others.
    - A name with no numeric suffix → scalar.

    Parameters
    ----------
    input_variables : list[str]
        All detected input variable names.

    Returns
    -------
    field3d_bases : list[str]
        Base names of 3D fields, e.g. ['gravity', 'magnetic'].
        Ordered as they appear (preserving expression-detection order).
    scalar_names : list[str]
        Names of scalar inputs, e.g. ['motor_torque', 'fan_force'].

    Raises
    ------
    ValueError
        If a field base has unexpected numeric suffixes.
    """

    # Split each variable into (base, suffix) where suffix is trailing digits or ""
    suffix_pattern = re.compile(r"^(.*?)(\d+)?$")

    # Group variables by their base name
    base_to_suffixes = {}
    for var in input_variables:
        match = suffix_pattern.match(var)
        base, suffix = match.group(1), match.group(2)
        base_to_suffixes.setdefault(base, set())
        if suffix is not None:
            base_to_suffixes[base].add(int(suffix))
        # suffix None means no digits at all → scalar, handled below

    field3d_bases = []
    scalar_names = []

    for var in input_variables:
        match = suffix_pattern.match(var)
        base, suffix = match.group(1), match.group(2)

        if suffix is None:
            # No numeric suffix → scalar, add once
            scalar_names.append(var)
        elif base not in [b for b in field3d_bases]:
            # First time we see this base → validate and register as field
            found_suffixes = base_to_suffixes[base]
            unexpected = found_suffixes - {0, 1, 2}
            if unexpected:
                raise ValueError(
                    f"Input '{base}' has unexpected component indices {sorted(unexpected)}. "
                    f"3D fields must only use suffixes 0, 1, 2."
                )
            field3d_bases.append(base)
        # else: base already registered, skip remaining components

    return field3d_bases, scalar_names


def _validate_inputs(sphere_data: list, input_variables: list[str], constants: dict):
    """
    Validate that input variables are used correctly across all sphere expressions.

    Rules enforced:
    1. Input variables must NOT appear in radius, position, or orientation expressions.
    2. Input variables must appear LINEARLY in force and torque expressions.

    Parameters
    ----------
    sphere_data : list
        Raw sphere data from YAML.
    input_variables : list[str]
        All input variable names in canonical order.
    constants : dict
        Constant substitutions to apply before checking.

    Raises
    ------
    ValueError
        If any input variable appears in geometry expressions, or appears
        nonlinearly in force/torque expressions.
    """
    input_symbols = [sp.Symbol(v) for v in input_variables]

    for i, data in enumerate(sphere_data):
        # --- Geometry expressions: inputs must be absent ---
        geometry_exprs = {
            "radius": [str(data.get("radius", 1))],
            "position": [str(x) for x in data.get("position", [0, 0, 0])],
            "orientation": [str(x) for x in data.get("orientation", [0, 0, 0])],
        }
        for field_name, exprs in geometry_exprs.items():
            for expr_str in exprs:
                expr = sp.sympify(expr_str).subs(constants)
                used_inputs = [s for s in input_symbols if s in expr.free_symbols]
                if used_inputs:
                    raise ValueError(
                        f"Sphere {i}, {field_name}: input variable(s) "
                        f"{[str(s) for s in used_inputs]} found in a geometry expression. "
                        f"Inputs may only appear in force and torque."
                    )

        # --- Force/torque expressions: inputs must appear linearly ---
        load_exprs = {
            "force": [str(x) for x in data.get("force", [0, 0, 0])],
            "torque": [str(x) for x in data.get("torque", [0, 0, 0])],
        }
        for field_name, exprs in load_exprs.items():
            for expr_str in exprs:
                expr = sp.sympify(expr_str).subs(constants)
                for sym in input_symbols:
                    if sym not in expr.free_symbols:
                        continue
                    if not sp.simplify(sp.diff(expr, sym, 2)).is_zero:
                        raise ValueError(
                            f"Sphere {i}, {field_name}: expression '{expr}' is nonlinear "
                            f"in input '{sym}'. All inputs must appear linearly."
                        )
