"""Assembly of spheres used to define a soft body."""

import os
import re
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
import yaml

from .sphere import Sphere


class SphereAssembly:
    """
    The class `SphereAssembly` represents an assembly of rigid spheres connected by springs used to
    define a soft body.

    There are three set of variables:

    - Degrees of freedom (`dofs`): variables that define the deformation of the assembly (e.g. lengths
      of springs)
    - Design variables (`design`): parameters that define the morphology of the assembly and that can
      be optimized (e.g. radius, stiffness)
    - Input variables (`input`): parameters that define the input fields, either 3D-fields (e.g. gravity,
      magnetic field) or scalars (e.g. internal active force)

    Supports initialization from a YAML parameters file or incremental construction via methods.
    Enables symbolic and numerical computation of coupling matrices, Jacobians, and forces.
    Compatible with JAX for automatic differentiation and just-in-time compilation.

    Parameters
    ----------
    parameters_source : str, optional
        Path to a YAML file containing assembly parameters (spheres, dofs, design variables, etc.).
        If provided, the assembly is initialized from this file.
    verbose : bool, default=True
        If True, prints debug and progress information during initialization and computation.

    Attributes
    ----------
    Ndof : int
        Number of degrees of freedom in the assembly.
    Ndesign : int
        Number of design variables in the assembly.
    Ninput : int
        Number of input variables in the assembly.
    Nspheres : int
        Number of spheres in the assembly.
    dof_variables : list of str
        Names of the degrees of freedom.
    design_variables : list of str
        Names of the design variables.
    input_variables : list of str
        Names of the input variables.
    dof_defaults : dict
        Default values for the degrees of freedom.
    design_defaults : dict
        Default values for the design variables.

    Examples
    --------
    Initialize an empty assembly:

    >>> assembly = SphereAssembly()

    Initialize from a YAML file:

    >>> assembly = SphereAssembly(parameters_source="path/to/parameters.yaml")

    Add a sphere and a degree of freedom:

    >>> sphere = Sphere(radius=1.0, position=[0, 0, 0])
    >>> assembly.add_sphere(sphere)
    >>> assembly.add_dof(name="rotation", default=0.0)

    Notes
    -----
    - The assembly supports numerical evaluation and automatic differentiation via JAX.
    - Input variables are classified into 3D fields and scalars based on naming conventions.
    - Design parameters has a fixed shape after construction for JAX compatibility.
    """

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
        self._refresh_derived_couplings()

    @classmethod
    def from_file(cls, parameters_source: str, verbose=True):
        """Alternative constructor to initialize from a YAML file."""
        return cls(parameters_source, verbose)

    def add_sphere(self, sphere: Sphere):
        """Add a Sphere object to the assembly."""
        if not sphere._has_explicit_couplings:
            self._set_sphere_couplings_from_force(sphere, self.Nspheres)

        # Check that the sphere has the correct number of degrees of freedom and parameters
        try:
            test_dof = jnp.zeros(self.Ndof)
            test_design = jnp.zeros(self.Ndesign)
            test_time = jnp.zeros((1,))

            sphere.radius(dofs=test_dof, design=test_design)
            sphere.position(dofs=test_dof, design=test_design, time=test_time)
            sphere.orientation(dofs=test_dof, design=test_design, time=test_time)
            C_H = sphere.C_H(dofs=test_dof, design=test_design)
            C_K = sphere.C_K(dofs=test_dof, design=test_design)

            if C_H.shape != (6, self.Ninput):
                raise ValueError(f"C_H must have shape (6, {self.Ninput}), but got {C_H.shape}.")
            if C_K.shape != (6, self.Ndof):
                raise ValueError(f"C_K must have shape (6, {self.Ndof}), but got {C_K.shape}.")

        except ValueError as e:
            raise ValueError(
                f"Sphere does not have the correct number of degrees of freedom or variables: {e}"
            ) from e

        self.spheres.append(sphere)
        self._Nspheres += 1  # Update sphere count

    def add_dof(self, name: str, default: float = 0.0):
        """Add a new degree of freedom (DOF)."""
        if name in self._dof_variables:
            raise ValueError(f"DOF '{name}' already exists.")

        self._dof_variables.append(name)
        self._Ndof += 1
        self._dof_defaults = jnp.append(self._dof_defaults, default)
        self._refresh_derived_couplings()
        print(f"NEW degrees of freedom\n {self.dof_variables} \nwith default values\n {self.dof_defaults}")

    def add_design(self, name: str, default: float = 0.0):
        """Add a new design parameter."""
        if name in self._design_variables:
            raise ValueError(f"Design parameter '{name}' already exists.")

        self._design_variables.append(name)
        self._Ndesign += 1
        self._design_defaults = jnp.append(self._design_defaults, default)
        self._refresh_derived_couplings()
        print(f"NEW design parameters\n {self.design_variables} \nwith default values\n {self.design_defaults}")

    def add_input(self, name: str, kind: str = "scalar"):
        """Add a scalar input or a three-component field input."""
        if not name:
            raise ValueError("Input name must be non-empty.")
        if kind not in {"scalar", "field"}:
            raise ValueError("Input kind must be 'scalar' or 'field'.")

        new_variables = [name] if kind == "scalar" else [f"{name}{i}" for i in range(3)]
        existing = set(self._input_variables)
        duplicates = [var for var in new_variables if var in existing]
        if duplicates:
            raise ValueError(f"Input variable(s) already exist: {duplicates}")

        if kind == "field":
            field_variables = [var for var in self._input_variables if var[-1].isdigit()]
            scalar_variables = [var for var in self._input_variables if not var[-1].isdigit()]
            self._input_variables = field_variables + new_variables + scalar_variables
        else:
            self._input_variables.extend(new_variables)
        self._Ninput += len(new_variables)
        self._input_defaults = jnp.zeros(self._Ninput)
        self._refresh_derived_couplings()
        print(f"NEW input parameters\n {self.input_variables}")

    # Read-only properties for fundamental attributes
    @property
    def Ndof(self):
        """Number of dynamic degrees of freedom."""
        return self._Ndof

    @property
    def Ndesign(self):
        """Number of design variables."""
        return self._Ndesign

    @property
    def Ninput(self):
        """Number of input variables."""
        return self._Ninput

    @property
    def Nspheres(self):
        """Number of spheres in the assembly."""
        return self._Nspheres

    @property
    def dof_variables(self):
        """Names of degrees of freedom in canonical order."""
        return self._dof_variables

    @property
    def design_variables(self):
        """Names of design variables in canonical order."""
        return self._design_variables

    @property
    def input_variables(self):
        """Names of input variables in canonical order."""
        return self._input_variables

    @property
    def dof_defaults(self):
        """Default degree-of-freedom values."""
        return self._dof_defaults

    @property
    def design_defaults(self):
        """Default design-variable values."""
        return self._design_defaults

    def grand_C_H(self, dofs=None, design=None, time=None):
        """
        Returns C_H of shape (6N, Ninput) such that:
            grand_force_torque = C_H @ inputs + C_K @ dofs
        """
        dofs, design, time = self._setup_params(dofs, design, time)
        return jnp.vstack([sphere.C_H(dofs, design) for sphere in self.spheres])

    def grand_C_K(self, dofs=None, design=None, time=None):
        """
        Returns C_K of shape (6N, Ndof) such that:
            grand_force_torque = C_H @ inputs + C_K @ dofs
        """
        dofs, design, time = self._setup_params(dofs, design, time)
        return jnp.vstack([sphere.C_K(dofs, design) for sphere in self.spheres])

    def _refresh_derived_couplings(self):
        """Refresh coupling functions for spheres defined by force/torque."""
        for i, sphere in enumerate(self.spheres):
            if not sphere._has_explicit_couplings:
                self._set_sphere_couplings_from_force(sphere, i)

    def _set_sphere_couplings_from_force(self, sphere: Sphere, sphere_index=None):
        """Derive C_H and C_K from the sphere's six-component force callable."""
        ninput = self.Ninput
        ndof = self.Ndof
        inputs_zero = jnp.zeros(ninput)
        dofs_ref = jnp.asarray(self.dof_defaults, dtype=float)
        design_ref = jnp.asarray(self.design_defaults, dtype=float)

        _validate_six_component_force_linearity(
            sphere,
            dofs_ref,
            design_ref,
            inputs_zero,
            sphere_index=sphere_index,
        )

        def C_H_func(dofs, design):
            if ninput == 0:
                return jnp.zeros((6, 0))
            try:
                return jax.jacfwd(
                    lambda inputs: sphere.six_component_force(dofs, design, inputs)
                )(jnp.zeros(ninput))
            except Exception:
                _dofs_np = np.asarray(dofs, dtype=float)
                _design_np = np.asarray(design, dtype=float)
                return _numerical_jacobian(
                    lambda inputs: _six_component_force_np64(sphere, _dofs_np, _design_np, inputs),
                    np.zeros(ninput),
                )

        def C_K_func(dofs, design):
            if ndof == 0:
                return jnp.zeros((6, 0))
            try:
                return jax.jacfwd(
                    lambda q: sphere.six_component_force(q, design, jnp.zeros(ninput))
                )(dofs)
            except Exception:
                _design_np = np.asarray(design, dtype=float)
                return _numerical_jacobian(
                    lambda q: _six_component_force_np64(sphere, q, _design_np, np.zeros(ninput)),
                    np.asarray(dofs, dtype=float),
                )

        C_H_func.expression = "d(six_component_force) / d(inputs)"
        C_K_func.expression = "d(six_component_force(dofs, design, 0)) / d(dofs)"
        sphere._set_coupling_functions(C_H_func, C_K_func)

    def compute_Jassembly(self, dofs=None, design=None, time=None):
        """
        Computes J_sph, which is defined by V = J_sph . dotQ, with V the grand velocity in the body's
        frame and dotQ the time derivative of the dofs.
        We use: V = B . dotX = B . J_X . dotQ, suh that J_sph = B . J_X

        Args:
            dofs (jnp.ndarray, optional): Degrees of freedom. Defaults to None.
            design (jnp.ndarray, optional): Design Parameters. Defaults to None.

        Returns:
            jnp_array: J_sph
        """
        dofs, design, time = self._setup_params(dofs, design, time)

        # Compute the Jacobian of X with respect to dofs Q using JAX's automatic differentiation
        Jacobian_X = jax.jacfwd(self._grand_coordinates_func, argnums=0)
        J_X = jnp.array(Jacobian_X(dofs, design, time))

        # Velocity: derivative of X with respect to time
        velocity = jax.jacfwd(self._grand_coordinates_func, argnums=2)
        v = jnp.array(velocity(dofs, design, time))

        # Create the block-diagonal
        # Each block is computed by bortz_jacobian_for_sphere for each sphere
        Bi_s = [sphere.bortz_jacobian(dofs, design, time) for sphere in self.spheres]

        # Construct N by stacking the blocks along the diagonal
        B = jnp.block(
            [
                [block if i == j else jnp.zeros_like(block) for j, block in enumerate(Bi_s)]
                for i in range(self.Nspheres)
            ]
        )

        J_sph = B @ J_X
        v_act = B @ v

        return J_sph, v_act

    def compute_C_U(self, dofs=None, design=None, time=None):
        """
        Computes C_U, such that v = V + C_U .v_0, with V and v the grand velocity in the body and lab
        frame, and v_0 the six-component velocity of the body reference in the lab frame

        Args:
            dofs (jnp.ndarray, optional): Degrees of freedom. Defaults to None.
            design (jnp.ndarray, optional): Design Parameters. Defaults to None.

        Returns:
            jnp.array: C_U
        """
        dofs, design, time = self._setup_params(dofs, design, time)

        # Construct T by assembling vertically the individual T's for each sphere
        C_U = jnp.block([[sphere.composition_of_velocity(dofs, design, time)] for sphere in self.spheres])

        return C_U

    def compute_composition_of_forces(self, dofs=None, design=None, time=None):
        """
        Compute the grand force-composition matrix.

        Parameters
        ----------
        dofs : array-like, optional
            Degrees of freedom. Defaults to ``dof_defaults``.
        design : array-like, optional
            Design variables. Defaults to ``design_defaults``.
        time : float or array-like, optional
            Time used for time-dependent geometry.

        Returns
        -------
        jnp.ndarray
            Block-diagonal matrix of shape ``(6*Nspheres, 6*Nspheres)`` that
            maps sphere forces and torques to the body-reference convention.
        """
        dofs, design, time = self._setup_params(dofs, design, time)

        # Create blocks for individual spheres
        blocks = [sphere.composition_of_force(dofs, design, time) for sphere in self.spheres]

        # Construct Tf by stacking blocks along the diagonal
        Tf = jnp.block(
            [[b if i == j else jnp.zeros_like(b) for j, b in enumerate(blocks)] for i in range(self.Nspheres)]
        )

        return Tf

    def compute_Jacobian_matrix(self, dofs=None, design=None, time=None):
        """
        Computes the Jacobian tensor J = partial v / partial p

        Args:
            dofs (jnp.ndarray, optional): Degrees of freedom. Defaults to None.
            design (jnp.ndarray, optional): Design Parameters. Defaults to None.

        Returns:
            jnp.array: J
        """
        dofs, design, time = self._setup_params(dofs, design, time)

        Jass, v_act = self.compute_Jassembly(dofs, design, time)
        C_U = self.compute_C_U(dofs, design, time)
        J = jnp.block([C_U, Jass])

        return J, v_act

    def set_dof_defaults(self, new_dofs=None, new_dict=None, verbose=True):
        """
        Sets new default values for the degrees of freedom (dofs).

        Args:
            new_dofs (array-like, optional): New default values; length must match Ndof.
            new_dict (dict, optional): Mapping of DOF variable names to new default values.
            verbose (bool, optional): If True, prints old and new defaults. Defaults to True.

        Raises:
            ValueError: If ``new_dofs`` has an invalid shape, or a name in ``new_dict``
                is not a known DOF.
        """
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
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid value for variable '{key}': {value}") from exc

                # Update the corresponding index in dof_defaults
                self._dof_defaults = self._dof_defaults.at[idx].set(new_value)
        if verbose:
            print("NEW default dof values:", self.dof_variables, self.dof_defaults)

    def set_design_defaults(self, new_design=None, new_dict=None, verbose=True):
        """
        Sets new default values for the design variables.

        Args:
            new_design (array-like, optional): New default values; length must match Ndesign.
            new_dict (dict, optional): Mapping of design variable names to new default values.
            verbose (bool, optional): If True, prints old and new defaults. Defaults to True.

        Raises:
            ValueError: If ``new_design`` has an invalid shape, or a name in ``new_dict``
                is not a known design variable.
        """
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
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid value for variable '{key}': {value}") from exc

                # Update the corresponding index in design_defaults
                self._design_defaults = self._design_defaults.at[idx].set(new_value)
        if verbose:
            print("NEW default param values:", self.design_variables, self.design_defaults)

    # def _grand_radius_func(self, dofs=None, design=None, time=None):
    #     dofs, design, time = self._setup_params(dofs, design, time)

    #     return jnp.concatenate([jnp.array([sphere.radius(dofs, design)]) for sphere in self.spheres])

    def _grand_coordinates_func(self, dofs=None, design=None, time=None):
        dofs, design, time = self._setup_params(dofs, design, time)
        coords = [
            jnp.concatenate([sphere.position(dofs, design, time), sphere.orientation(dofs, design, time)])
            for sphere in self.spheres
        ]
        return jnp.concatenate(coords)

    def __str__(self):
        return (
            f"Assembly with {self.Nspheres} spheres, {self.Ndof} degrees of freedom, "
            f"and {self.Ndesign} fixed parameters"
        )

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
            output += f"  position: {sphere.position(self.dof_defaults, self.design_defaults, [0.0])}\n"
            output += f"  orientation: {sphere.orientation(self.dof_defaults, self.design_defaults, [0.0])}\n"
            output += f"  C_H:\n{sphere.C_H(self.dof_defaults, self.design_defaults)}\n"
            output += f"  C_K:\n{sphere.C_K(self.dof_defaults, self.design_defaults)}\n"

        return output

    def _setup_params(self, dofs=None, design=None, time=None):
        if dofs is None:
            dofs = self.dof_defaults
        if design is None:
            design = self.design_defaults
        if time is None:
            time = 0.0

        dofs = jnp.atleast_1d(jnp.asarray(dofs, dtype=float))
        design = jnp.atleast_1d(jnp.asarray(design, dtype=float))
        time = jnp.atleast_1d(jnp.asarray(time, dtype=float))

        return dofs, design, time

    def _parse_parameter_file(self, parameters_source: str, verbose: bool):
        # Read YAML data from file or string
        try:
            # Check if the input is a file path (ends with .yaml or .yml, case insensitive)
            if parameters_source.lower().endswith((".yaml", ".yml")):
                if not os.path.exists(parameters_source):
                    raise FileNotFoundError(f"File not found: {parameters_source}")
                with open(parameters_source) as f:
                    yaml_data = yaml.safe_load(f)
                    if verbose:
                        print(f"Parsing parameter file: {parameters_source}")
            else:
                yaml_data = yaml.safe_load(StringIO(parameters_source))

            if not isinstance(yaml_data, dict):
                raise ValueError("Invalid YAML format: Expected a dictionary at the root level.")

        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            raise ValueError(f"Error loading parameters: {e}") from e

        # Extract sphere data
        sphere_data = yaml_data["spheres"]

        # Extract all possible prefixes from dof_names and design_names
        dof_prefixes = {"dof"}
        design_prefixes = {"design"}
        input_prefixes = {"input"}

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

            DIGIT_SUFFIX = r"(?:\d+)?"

            dof_patterns = [re.compile(r"\b(?:" + p + r")" + DIGIT_SUFFIX + r"\b") for p in dof_prefixes]
            design_patterns = [re.compile(r"\b(?:" + p + r")" + DIGIT_SUFFIX + r"\b") for p in design_prefixes]
            input_patterns = [re.compile(r"\b(?:" + p + r")" + DIGIT_SUFFIX + r"\b") for p in input_prefixes]

            for expr in sphere_exprs:
                for pat in dof_patterns:
                    dof_set.update(pat.findall(expr))
                for pat in design_patterns:
                    design_set.update(pat.findall(expr))
                for pat in input_patterns:
                    input_set.update(pat.findall(expr))

            for expr in sphere_exprs:
                all_detected_symbols.update(_free_symbol_names(expr))

        # Transform variable sets into lists and sort them alphabetically
        dof_variables = sorted(list(dof_set))
        design_variables = sorted(list(design_set))
        input_variables = sorted(list(input_set))
        time_variable = ["time"]

        # Classify and validate input variable names
        field3d_bases, scalar_names = _classify_input_variables(input_variables)

        # Reorder canonically: field components first (in field order), scalars last
        input_variables = [base + str(j) for base in field3d_bases for j in range(3)] + scalar_names
        num_input = len(input_variables)

        # printing variables found
        if verbose:
            dof_part = ", ".join(dof_variables) + ", " if dof_variables else ""
            design_part = ", ".join(design_variables) + ", " if design_variables else ""
            input_part = ", ".join(input_variables)
            print(f"  Found variables: {dof_part}{design_part}{input_part}")
            if field3d_bases:
                print(f"  3D field inputs:  {field3d_bases}")
            if scalar_names:
                print(f"  Scalar inputs:    {scalar_names}")

        # Determine Ndof, Ndesign, and Nsphere
        num_dof = len(dof_variables)
        num_design = len(design_variables)
        num_input = len(input_variables)
        num_sphere = len(sphere_data)

        # Create default arrays with zeros
        dof_defaults = jnp.zeros(num_dof)
        design_defaults = jnp.zeros(num_design)

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
        known_symbols = (
            set(dof_variables)
            | set(design_variables)
            | set(input_variables)
            | set(time_variable)
            | set(constants.keys())
        )

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

            rad_func = _create_function(rad_exprs, dof_variables, design_variables, constants)
            pos_func = _create_function_time(pos_exprs, dof_variables, design_variables, constants, time_variable)
            ori_func = _create_function_time(ori_exprs, dof_variables, design_variables, constants, time_variable)
            force_func = _create_input_function(
                for_exprs, dof_variables, design_variables, input_variables, constants
            )
            torque_func = _create_input_function(
                tor_exprs, dof_variables, design_variables, input_variables, constants
            )
            spheres.append(Sphere(rad_func, pos_func, ori_func, force=force_func, torque=torque_func))

            # Printing the characteristics of each sphere
            if verbose:
                print(f"    Sphere {i}")
                print(f"      Radius: {rad_func.expression}")
                print(f"      Position: {pos_func.expression}")
                print(f"      Orientation: {ori_func.expression}")
                print(f"      Force: {force_func.expression}")
                print(f"      Torque: {torque_func.expression}")

        return (
            num_dof,
            num_design,
            num_input,
            num_sphere,
            dof_variables,
            design_variables,
            input_variables,
            dof_defaults,
            design_defaults,
            spheres,
        )


# Useful functions ###########################################################

JAX_MODULES = {
    "jax": jnp,
    "array": jnp.array,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "abs": jnp.abs,
    "pi": jnp.pi,
    "arcsin": jnp.arcsin,
    "arccos": jnp.arccos,
    "arctan": jnp.arctan,
    "arctan2": jnp.arctan2,
}


# Caches for SymPy parsing.  Large assemblies often re-parse the same handful of
# expression strings ("0", "radius", "k * x0", …); without caching the cost
# scales linearly in N × (#expressions per sphere).  Keys are plain tuples of
# strings so they are cheap to hash.
_sympify_cache: dict = {}
_lambdify_cache: dict = {}


def _cached_sympify_subs(expr_str: str, constants: dict) -> sp.Expr:
    """Return ``sp.sympify(expr_str).subs(constants)`` with memoisation."""
    key = (expr_str, tuple(sorted(constants.items())))
    cached = _sympify_cache.get(key)
    if cached is None:
        cached = sp.sympify(expr_str).subs(constants)
        _sympify_cache[key] = cached
    return cached


def _free_symbol_names(expr_str: str) -> set:
    """Return free-symbol names of ``expr_str`` with memoisation (no substitution)."""
    key = (expr_str, ())
    cached = _sympify_cache.get(key)
    if cached is None:
        cached = sp.sympify(expr_str)
        _sympify_cache[key] = cached
    return {str(s) for s in cached.free_symbols}


def _cached_lambdify(sp_expr: sp.Expr, var_groups: tuple, modules=JAX_MODULES):
    """Memoised ``sp.lambdify(var_groups, sp_expr, modules)``.

    ``var_groups`` is a tuple of tuples of symbol names — e.g.
    ``(("x0",), ("k",), ("time",))`` — describing the grouped argument
    structure of the lambdified function.
    """
    key = (str(sp_expr), var_groups)
    cached = _lambdify_cache.get(key)
    if cached is not None:
        return cached
    symbol_groups = [[sp.symbols(name) for name in group] for group in var_groups]
    func = sp.lambdify(symbol_groups, sp_expr, modules)
    _lambdify_cache[key] = func
    return func


def _create_function(sp_exprs, dofs, design, constants):
    """
    Create a function that takes dofs and yaml_data as input and returns the evaluated symbolic
    expressions sp_exprs.

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
    var_groups = (tuple(dofs), tuple(design))

    if isinstance(sp_exprs, str):
        sp_expr = _cached_sympify_subs(sp_exprs, constants)

        try:
            jax_expr = _cached_lambdify(sp_expr, var_groups)
        except Exception as e:
            raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}") from e

        def wrapper(dof_args, design_args):
            result = jax_expr(dof_args, design_args)
            return jnp.asarray(result, dtype=float).reshape(())

        wrapper.expression = str(sp_expr)
        return wrapper
    else:
        sp_exprs = [_cached_sympify_subs(expr, constants) for expr in sp_exprs]
        jax_exprs = []

        for sp_expr in sp_exprs:
            try:
                jax_exprs.append(_cached_lambdify(sp_expr, var_groups))
            except Exception as e:
                raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}") from e

        def wrapper(dof_args, design_args):
            return jnp.stack(
                [jnp.asarray(jax_expr(dof_args, design_args), dtype=float).reshape(()) for jax_expr in jax_exprs]
            )

        wrapper.expression = [str(sp_expr) for sp_expr in sp_exprs]
        return wrapper


def _create_function_time(sp_exprs, dofs, design, constants, time):
    var_groups = (tuple(dofs), tuple(design), tuple(time))

    if isinstance(sp_exprs, str):
        sp_expr = _cached_sympify_subs(sp_exprs, constants)

        try:
            jax_expr = _cached_lambdify(sp_expr, var_groups)
        except Exception as e:
            raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}") from e

        def wrapper(dof_args, design_args, time_args):
            result = jax_expr(dof_args, design_args, time_args)
            return jnp.asarray(result, dtype=float).reshape(())

        wrapper.expression = str(sp_expr)
        return wrapper
    else:
        sp_exprs = [_cached_sympify_subs(expr, constants) for expr in sp_exprs]
        jax_exprs = []

        for sp_expr in sp_exprs:
            try:
                jax_exprs.append(_cached_lambdify(sp_expr, var_groups))
            except Exception as e:
                raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}") from e

        def wrapper(dof_args, design_args, time_args):
            return jnp.stack(
                [
                    jnp.asarray(jax_expr(dof_args, design_args, time_args), dtype=float).reshape(())
                    for jax_expr in jax_exprs
                ]
            )

        wrapper.expression = [str(sp_expr) for sp_expr in sp_exprs]
        return wrapper


def _create_input_function(sp_exprs, dofs, design, input_variables, constants):
    """Create a callable that evaluates expressions using dofs, design, and inputs."""
    var_groups = (tuple(dofs), tuple(design), tuple(input_variables))

    if isinstance(sp_exprs, str):
        sp_expr = _cached_sympify_subs(sp_exprs, constants)
        try:
            jax_expr = _cached_lambdify(sp_expr, var_groups)
        except Exception as e:
            raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}") from e

        def wrapper(dof_args, design_args, input_args):
            result = jax_expr(dof_args, design_args, input_args)
            return jnp.asarray(result, dtype=float).reshape(())

        wrapper.expression = str(sp_expr)
        return wrapper

    sp_exprs = [_cached_sympify_subs(expr, constants) for expr in sp_exprs]
    jax_exprs = []

    for sp_expr in sp_exprs:
        try:
            jax_exprs.append(_cached_lambdify(sp_expr, var_groups))
        except Exception as e:
            raise ValueError(f"Error converting expression {sp_expr} with sympy.lambdify: {e}") from e

    def wrapper(dof_args, design_args, input_args):
        return jnp.stack(
            [
                jnp.asarray(jax_expr(dof_args, design_args, input_args), dtype=float).reshape(())
                for jax_expr in jax_exprs
            ]
        )

    wrapper.expression = [str(sp_expr) for sp_expr in sp_exprs]
    return wrapper


def _six_component_force_np64(sphere, dofs, design, inputs):
    """Evaluate force+torque returning numpy float64, bypassing JAX float32 coercion.

    Uses the raw user callable stored on the force/torque function when available,
    so float64 precision is preserved end-to-end for numerical differentiation.
    """
    def _eval_raw(func, *args):
        raw = getattr(func, "_raw", None)
        if raw is not None:
            return np.asarray(raw(*args), dtype=np.float64)
        return np.asarray(func(*args), dtype=np.float64)

    f = _eval_raw(sphere._force_func, dofs, design, inputs)
    t = _eval_raw(sphere._torque_func, dofs, design, inputs)
    return np.concatenate([f, t])


def _numerical_jacobian(func, x, eps: float = 1e-7) -> jnp.ndarray:
    """Central-difference Jacobian. Calls func with concrete numpy arrays."""
    x_np = np.asarray(x, dtype=float)
    n = x_np.shape[0]
    f0 = np.asarray(func(x_np), dtype=float)
    m = f0.shape[0]
    if n == 0:
        return jnp.zeros((m, 0))
    cols = []
    for i in range(n):
        xp, xm = x_np.copy(), x_np.copy()
        xp[i] += eps
        xm[i] -= eps
        cols.append((np.asarray(func(xp), dtype=float) - np.asarray(func(xm), dtype=float)) / (2.0 * eps))
    return jnp.asarray(np.column_stack(cols))


def _validate_six_component_force_linearity(
    sphere: Sphere,
    dofs,
    design,
    inputs_zero,
    sphere_index=None,
    atol=1e-5,
    rtol=1e-5,
):
    """Validate the shape and sampled input-linearity of a six-component force.

    Tolerances are sized for JAX's default float32 (≈ 1e-7 relative) plus the
    finite-difference epsilon used in :func:`_numerical_jacobian` (≈ 1e-7),
    which can compound to a few×1e-7. ``atol``/``rtol`` are therefore set to
    ``1e-5`` — still much tighter than any genuine nonlinearity of practical
    interest.
    """
    ninput = inputs_zero.shape[0]
    prefix = f"Sphere {sphere_index}: " if sphere_index is not None else ""
    dofs_np = np.asarray(dofs, dtype=float)
    design_np = np.asarray(design, dtype=float)

    def force_for_inputs(inputs):
        value = _six_component_force_np64(sphere, dofs_np, design_np, np.asarray(inputs, dtype=float))
        if value.shape != (6,):
            raise ValueError(f"{prefix}six_component_force must have shape (6,), but got {value.shape}.")
        return value

    base = force_for_inputs(np.asarray(inputs_zero, dtype=float))
    if ninput == 0:
        return

    test_inputs = [np.ones(ninput), np.arange(1, ninput + 1, dtype=float) / max(ninput, 1)]

    try:
        # Exact check via forward-mode AD (works for JAX-native callables).
        def jax_force(inputs):
            return jnp.asarray(sphere.six_component_force(dofs_np, design_np, inputs), dtype=float)

        C_H_jax = jax.jacfwd(jax_force)(jnp.zeros(ninput))
        for test_input in test_inputs:
            second = jax.jacfwd(jax.jacfwd(jax_force))(jnp.asarray(test_input))
            if not bool(jnp.allclose(second, jnp.zeros_like(second), atol=atol, rtol=rtol)):
                raise ValueError(f"{prefix}force/torque must be linear in inputs to derive C_H.")
            expected = base + np.asarray(C_H_jax) @ test_input
            actual = force_for_inputs(test_input)
            if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                raise ValueError(f"{prefix}force/torque must be linear in inputs to derive C_H.")
    except ValueError:
        raise
    except Exception:
        # Callable uses non-JAX ops; check linearity numerically.
        C_H_num = np.asarray(_numerical_jacobian(force_for_inputs, np.zeros(ninput)))
        for test_input in test_inputs:
            expected = base + C_H_num @ test_input
            actual = force_for_inputs(test_input)
            if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                raise ValueError(f"{prefix}force/torque must be linear in inputs to derive C_H.") from None


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
                expr = _cached_sympify_subs(expr_str, constants)
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
                expr = _cached_sympify_subs(expr_str, constants)
                for sym in input_symbols:
                    if sym not in expr.free_symbols:
                        continue
                    if not sp.simplify(sp.diff(expr, sym, 2)).is_zero:
                        raise ValueError(
                            f"Sphere {i}, {field_name}: expression '{expr}' is nonlinear "
                            f"in input '{sym}'. All inputs must appear linearly."
                        )
