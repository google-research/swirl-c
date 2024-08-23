"""A library of boundary conditions enforced in a parallel setting."""

from typing import Dict, Tuple, Union
import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.common import types
from swirl_c.common import utils
from swirl_c.core import parameter
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
import tensorflow as tf

_DIMS = (0, 1, 2)
_REPLICA_DIMS = (0, 1, 2)

_CELL_FACE_BC = bc_types.BoundaryDictionaryType.CELL_FACES.value
_CELL_AVG_BC = bc_types.BoundaryDictionaryType.CELL_AVERAGES.value
_CELL_FLUX_BC = bc_types.BoundaryDictionaryType.INTERCELL_FLUXES.value


def exchange_halos(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    u: types.FlowFieldVar,
    bc_dict: bc_types.BCDict,
    width: int,
) -> types.FlowFieldVar:
  """Updates a flow field variable from specified boundary conditions.

  This is a wrapper for the Swirl-LM halo_exchange.inplace_halo_exchange.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    u: The flow field variable to be updated by the boundary condition.
    bc_dict: Dictionary of boundary conditions for the variable `u`
    width: An integer indicating the width of the halo region.

  Returns:
    The flow field variable `u` with the halo regions updated according to the
    boundary conditions and inter-replica exchange.
  """

  periodic_dims = [False, False, False]
  bc = [None, None, None]

  for dim, vals in bc_dict.items():
    buf = []
    for val in vals.values():
      bc_type = val[0]
      if bc_type == bc_types.BoundaryCondition.PERIODIC:
        periodic_dims[types.DIMS.index(dim)] = True
        buf = [None, None]
        break
      elif bc_type == bc_types.BoundaryCondition.DIRICHLET:
        buf.append((halo_exchange.BCType.DIRICHLET, val[1]))
      elif bc_type == bc_types.BoundaryCondition.NEUMANN:
        buf.append((halo_exchange.BCType.NEUMANN, val[1]))
      else:
        raise NotImplementedError(f'Unsupported BC type {bc_type}.')
    bc[types.DIMS.index(dim)] = buf

  return halo_exchange.inplace_halo_exchange(
      u,
      _DIMS,
      replica_id,
      replicas,
      _REPLICA_DIMS,
      periodic_dims,
      bc,
      width,
  )


def update_fluxes(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    fluxes: Dict[str, types.FlowFieldMap],
    cfg: parameter.SwirlCParameters,
    flux_type: bc_types.BoundaryFluxType,
) -> Dict[str, types.FlowFieldMap]:
  """Updates the fluxes from an imposed boundary condition.

  Only the flux of conservative flow field variables can be specified by the
  boundary condition, and only the "DIRICHLET" and `None` boundary type are
  supported.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    fluxes: A dictionary of flow field variables where each key is the direction
      of the flux and the value is a dictionary of flow field variables
      representing the flux of the conservative variables along the given
      direction.
    cfg: The context object that stores parameters and information required by
      the simulation.
    flux_type: The particular flux which the boundary condition is being applied
      for. Either `CONVECTIVE`, `DIFFUSIVE`, or `TOTAL`.

  Returns:
    A dictionary of flow field variables where each key, value pair refers to a
    flux along the given dim, equivalent to the arguement `fluxes`. The fluxes
    have been updated according to the boundary conditions.

  Raises:
    `ValueError` if the variable name in the boundary condition is not a
    conservative variable name associated with one of the flow field variables.
  """
  if (
      _CELL_FLUX_BC not in cfg.bc
      or flux_type.value not in cfg.bc[_CELL_FLUX_BC]
  ):
    return fluxes

  bc_var_names = set(cfg.bc[_CELL_FLUX_BC][flux_type.value])
  cons_var_names = set(cfg.conservative_variable_names)
  invalid_var_names = bc_var_names - cons_var_names
  assert not invalid_var_names, (
      f'{invalid_var_names} cannot be specified by a flux boundary condition.'
      f' Only conservative variables {cons_var_names} can be specified with a'
      ' flux boundary condition.'
  )

  def update_flux_1d(fluxes, dim, core_idx, plane_idx, bc_dict_1d):
    """Updates the flux values on 1 global plane."""
    if not bc_dict_1d:
      return fluxes
    for var_name, val in bc_dict_1d.items():
      fluxes[var_name] = common_ops.tensor_scatter_1d_update_global(
          replica_id,
          replicas,
          fluxes[var_name],
          types.DIMS.index(dim),
          core_idx,
          plane_idx,
          val,
      )
    return fluxes

  # When updating flux values for the various flux boundary condition, we
  # only update the fluxes which are across faces on the exterior of the
  # computational domain, i.e. the faces between a halo cell and a domain cell
  # on an exterior replica (processor). Recalling that the flux value stored at
  # i is the flux across the i - 1/2 face, the leftmost flux is stored in the
  # first "interior" mesh point, where the rightmost flux is the first "halo"
  # point.
  for dim in types.DIMS:
    for face in range(2):
      core_idx = 0 if face == 0 else replicas.shape[types.DIMS.index(dim)] - 1
      if face == 0:
        plane_idx = cfg.halo_width
      else:
        plane_idx = (
            cfg.halo_width
            + (cfg.core_nx, cfg.core_ny, cfg.core_nz)[types.DIMS.index(dim)]
        )

      bc_dict_1d = _get_bc_dict_along_dim(
          cfg.bc[_CELL_FLUX_BC][flux_type.value], dim, face
      )
      fluxes[dim] = update_flux_1d(
          fluxes[dim],
          dim,
          core_idx,
          plane_idx,
          bc_dict_1d,
      )
  return fluxes


def _get_bc_dict_along_dim(
    bc_dict: Dict[str, bc_types.BCDict],
    dim: str,
    side: int,
) -> Union[Dict[str, Union[types.FlowFieldVar, float]], None]:
  """Parses a dictionary of BC dictionaries along 1 dimension.

  This method extracts the boundary condition along a given dimension ('x', 'y',
  or 'z') on the negative (left, 0) or positive (right, 1) side of the domain
  for all variables whose boundary condition is specified.

  Args:
    bc_dict: A dictionary of boundary condiditon dictionaries, where each key,
      value pair refers to the boundary condition for a specified variable.
    dim: The dimension of the boundary to be extracted.
    side: An integer indicating the side of the domain to extract the BC for.
      `side = 0` indicates the negative (left) boundary along `dim`, and `side =
      1` indicates the positive (right) boundary along `dim`.

  Returns:
    A dictionary of the extracted boundary conditions, where keys are the
    variable specified by the boundary condition and values is the boundary
    value (either a `float` or a `tf.Tensor` represented as a single 2D tensor
    or a list of 2D tensors depending on `dim`) specified by the boundary
    condition.

  Raises:
    AssertionError: if the specified boundary condition is not `None` or
      "DIRICHLET".
  """

  bc = {}
  for var_name, var_bc_dict in bc_dict.items():
    bc_type, val = var_bc_dict[dim][side]
    if bc_type is None:
      continue
    assert bc_type is bc_types.BoundaryCondition.DIRICHLET, (
        f'Unsupported interface (face or flux) BC type {bc_type}. Only'
        ' "DIRICHLET" or `None` is supported.'
    )
    bc[var_name] = val
  return bc


def _face_boundary_update_1d(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    conservative: types.FlowFieldMap,
    primitive: types.FlowFieldMap,
    bc_dict_1d: Dict[str, Union[types.FlowFieldVar, float]],
    dim: str,
    core_index: int,
    plane_index: int,
    cfg: parameter.SwirlCParameters,
) -> Tuple[types.FlowFieldMap, types.FlowFieldMap]:
  """Updates conservative and primitive face values for one boundary.

  Note that this method only updates a single plane along `dim`, specified by
  `plane_index`. If two faces along `dim` must be updated (e.g. channel flow),
  then the method must be called for each face seperately.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    conservative: A dictionary of flow field variables representing the face
      intepolated conservative variables on the i - 1/2 face normal to `dim`.
    primitive: A dictionary of flow field variables representing the face
      intepolated primitive variables on the i - 1/2 face normal to `dim`.
    bc_dict_1d: A dictionary key, value pairs, where keys are the variable
      (conservative or primitive) to be specified by the boundary condition and
      values are the value of the variable on the boundary.
    dim: The dimension normal to the face where the boundary condition is to be
      applied.
    core_index: The index of the core in `dim`, in which the plane will be
      updated. The 3D tensor with other indices will remain unchanged.
    plane_index: The local index of the plane to be updated in `dim`.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A tuple `(conservative, primitive)` where `conservative` is the conservative
    flow field variables after the boundary update, and `primitive` is the
    associated primitive flow field variables after the boundary update.
  """

  del cfg  # Unused.

  # Ensure face BC is set along current dim.
  if not bc_dict_1d:
    return (conservative, primitive)

  # Treat density first to ensure subsequent primitive to conservative
  # conversions are correct. Update all primitives based on the new density.
  if types.RHO in bc_dict_1d:
    conservative[types.RHO] = common_ops.tensor_scatter_1d_update_global(
        replica_id,
        replicas,
        conservative[types.RHO],
        types.DIMS.index(dim),
        core_index,
        plane_index,
        bc_dict_1d[types.RHO],
    )
    # TODO: b/305817081 - Updating all primitives here is inefficient, and only
    # the primitives which are not updated by their associated conservative or
    # primitive boundary condition need to be updated. Filtering this call
    # to only those primitives would improve performance.
    primitive = utils.conservative_to_primitive_variables(conservative)

  for var_name, val in bc_dict_1d.items():
    if var_name is types.RHO:
      continue
    elif var_name in conservative:
      cons_var_name = var_name
      prim_var_name = types.conservative_to_primitive_name(var_name)
      conservative[cons_var_name] = common_ops.tensor_scatter_1d_update_global(
          replica_id,
          replicas,
          conservative[cons_var_name],
          types.DIMS.index(dim),
          core_index,
          plane_index,
          val,
      )
      primitive[prim_var_name] = tf.nest.map_structure(
          tf.math.divide_no_nan,
          conservative[cons_var_name],
          conservative[types.RHO],
      )
    else:
      if var_name not in primitive:
        raise ValueError(
            f'"{var_name}" is not a valid primitive or conservative variable'
            ' name for the face boundary condition update.'
        )
      prim_var_name = var_name
      cons_var_name = types.primitive_to_conservative_name(var_name)
      primitive[prim_var_name] = common_ops.tensor_scatter_1d_update_global(
          replica_id,
          replicas,
          primitive[prim_var_name],
          types.DIMS.index(dim),
          core_index,
          plane_index,
          val,
      )
      conservative[cons_var_name] = tf.nest.map_structure(
          tf.multiply, conservative[types.RHO], primitive[prim_var_name]
      )

  return (conservative, primitive)


def update_faces(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    conservative_neg: Dict[str, types.FlowFieldMap],
    conservative_pos: Dict[str, types.FlowFieldMap],
    primitive_neg: Dict[str, types.FlowFieldMap],
    primitive_pos: Dict[str, types.FlowFieldMap],
    cfg: parameter.SwirlCParameters,
) -> Tuple[
    Dict[str, types.FlowFieldMap],
    Dict[str, types.FlowFieldMap],
    Dict[str, types.FlowFieldMap],
    Dict[str, types.FlowFieldMap],
]:
  """Updates primitive and conservative face interpolated values from BCs.

  This method updates the face interpolated conservative and primitive flow
  field variables based on boundary conditions specified in `cfg.bc['faces']`
  . The only supported face boundary conditions are `DIRICHLET` or `None`. This
  boundary condition is assumed to be an external boundary condition, and thus
  only a single global plane is updated for a given dimension and side (e.g.
  `'x':0`). Only faces adjacent to the physical domain are updated (i.e. faces
  between halo and physical cells on exterior replicas). Faces between halo
  cells and interior faces between replicas are unchanged. Thus, at maximum 6
  global planes are updated here.

  Note that the method only supports setting a single face value at the boundary
  and does not support setting different positive and negative face
  interpolation values at the boundary.

  Currently the only variables supported are the conservative variable (`RHO`,
  `MOMENTUM`, `RHO_E`, and any product of a transported scalar and density
  `RHO`), and the associated primitive variables (`VELOCITY`, `E`, any
  additional transported scalars). See b/304330269 for more details.

  Args:
    replica_id: The ID of the computational subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    conservative_neg: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of
      conservative flow field variables interpolated onto the left (i - 1/2)
      face of the computational cell normal to the indicated dimension. The
      interpolation is performed with a negative stencil. All variables in
      `cfg.conservative_variable_names` are required, which at minimum will
      contain the density `RHO`, three components of momentum `RHO_U` `RHO_V`
      and `RHO_W`, the total energy `RHO_E`, and any additional transported
      scalars.
    conservative_pos: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of
      conservative flow field variables interpolated onto the left (i - 1/2)
      face of the computational cell normal to the indicated dimension. The
      interpolation is performed with a positive stencil. `conservative_pos`
      requires the same variables as `conservative_neg`
    primitive_neg: A dictionary with keys 'x', 'y', and 'z' representing each of
      the three directions, where each item contains a dictionary of primitive
      flow field variables interpolated onto the left (i - 1/2) face of the
      computational cell normal to the indicated dimension. The interpolation is
      performed with a negative stencil. In additional to the variables listed
      in `cfg.primitive_variable_names`, both the pressure `P` and total
      enthalpy `H` are required. `cfg.primitive_variable_names` must include at
      minimum the density `RHO`, three components of velocity `U`, `V`, `W`, the
      total energy `E`, and any additional transported scalars.
    primitive_pos: A dictionary with keys 'x', 'y', and 'z' representing each of
      the three directions, where each item contains a dictionary of primitive
      flow field variables interpolated onto the left (i - 1/2) face of the
      computational cell normal to the indicated dimension. The interpolation is
      performed with a positive stencil. `primitive_pos` requires the same
      variables asl `primitive_neg`.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A tuple `(conservative_neg, conservative_pos, primitive_neg, primitive_pos)`
    where `conservative_neg`, `conservative_pos`, `primitive_neg`, and
    `primitive_pos` are defined the same as their respective arguement, except
    now updated by the face boundary conditions.
  """
  if _CELL_FACE_BC not in cfg.bc:
    return (conservative_neg, conservative_pos, primitive_neg, primitive_pos)

  # When updating face interpolated values for the face boundary condition, we
  # only update the faces which are on the exterior of the computational domain,
  # i.e. the faces between a halo cell and a domain cell on an exterior
  # replica (processor). Recalling that the face value stored at i is the
  # i - 1/2 face, the leftmost face stored in the first "interior" mesh point,
  # where the rightmost face is the first "halo" point.
  for dim in types.DIMS:
    for face in range(2):
      core_idx = 0 if face == 0 else replicas.shape[types.DIMS.index(dim)] - 1
      if face == 0:
        plane_idx = cfg.halo_width
      else:
        plane_idx = (
            cfg.halo_width
            + (cfg.core_nx, cfg.core_ny, cfg.core_nz)[types.DIMS.index(dim)]
        )
      bc_dict_1d = _get_bc_dict_along_dim(cfg.bc[_CELL_FACE_BC], dim, face)
      conservative_neg[dim], primitive_neg[dim] = _face_boundary_update_1d(
          replica_id,
          replicas,
          conservative_neg[dim],
          primitive_neg[dim],
          bc_dict_1d,
          dim,
          core_idx,
          plane_idx,
          cfg,
      )
      conservative_pos[dim], primitive_pos[dim] = _face_boundary_update_1d(
          replica_id,
          replicas,
          conservative_pos[dim],
          primitive_pos[dim],
          bc_dict_1d,
          dim,
          core_idx,
          plane_idx,
          cfg,
      )

  return (conservative_neg, conservative_pos, primitive_neg, primitive_pos)


def update_conservative_cell_averages(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    conservative: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
) -> types.FlowFieldMap:
  """Updates the conservative cell average values from boundary condition (BC).

  This function will update the cell average values in the halo cells based on
  the boundary conditions specified in `cfg.bc['cells']`. Currently the only
  variables supported are the conservative variable (`RHO`, `MOMENTUM`, `RHO_E`,
  and any product of a transported scalar and density `RHO`), and the associated
  primitive variables (`VELOCITY`, `E`, any additional transported scalars). See
  b/304330269 for more details.

  Additionally, boundary conditions for a given flow field variable (e.g.
  x momentum `types.RHO_U`) must be given by the same variable for all three
  directions. That is, mixing `types.RHO_U` and `types.U` boundary dictionaries
  is not allowed.

  Note that this boundary update begins with `types.RHO` and proceeds in order
  through `conservative`. Therefore, it is possible that certain combinations of
  boundary condtions may not behave as intended due to order dependent updates
  to `conservative`.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    conservative: A dictionary of flow field variables representing the average
      value of conservative variables in the computational cell.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A dictionary of conservative flow field variables representing the cell
    average conservative quantities after update by the boundary conditions.

  Raises:
    `ValueError` if a "cells" boundary condition is not specified for a
    conservative variable, either directly or through the associated primitive
    variable.
  """
  for var_name, val in conservative.items():
    if var_name in cfg.bc[_CELL_AVG_BC]:
      conservative[var_name] = exchange_halos(
          replica_id,
          replicas,
          val,
          cfg.bc[_CELL_AVG_BC][var_name],
          cfg.halo_width,
      )
    elif types.conservative_to_primitive_name(var_name) in cfg.bc[_CELL_AVG_BC]:
      prim_buf = tf.nest.map_structure(
          tf.math.divide_no_nan,
          conservative[var_name],
          conservative[types.RHO],
      )
      prim_buf = exchange_halos(
          replica_id,
          replicas,
          prim_buf,
          cfg.bc[_CELL_AVG_BC][types.conservative_to_primitive_name(var_name)],
          cfg.halo_width,
      )
      conservative[var_name] = tf.nest.map_structure(
          tf.multiply, conservative[types.RHO], prim_buf
      )
    else:
      raise ValueError(
          'A "cells" boundary condition is required for all conservative'
          ' variables, either directly or through their respective primitive.'
          f' No boundary condition was found for {var_name} or'
          f' {types.conservative_to_primitive_name(var_name)}.'
      )
  return conservative


def _conservative_to_boundary_states(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    conservative: types.FlowFieldMap,
    bc_dict: bc_types.BCDict,
    cfg: parameter.SwirlCParameters,
) -> types.FlowFieldMap:
  """Converts conservative flow field variables to variables specified by BC.

  The provided conservative variables are cross refrenced with the variables
  specified in `bc_dict`, and where necessary conservative variables are
  converted to primitive variables. Currently the only variables supported are
  the conservative variable (`RHO`, `MOMENTUM`, `RHO_E`, and any product of a
  transported scalar and density `RHO`), and the associated primitive variables
  (`VELOCITY`, `E`, any additional transported scalars). See b/304330269 for
  more details.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    conservative: A dictionary of flow field variables representing the average
      value of conservative variables in the computational cell.
    bc_dict: The dictionary of the current boundary conditions being applied.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A dictionary of flow field variables which is a combination of conservative
    or primitive variables to be updated by the boundary conditions.

  Raises:
    `ValueError` if an inappropriate variable name is provided in `bc_dict`.
  """
  del replica_id, replicas, cfg

  states = {types.RHO: conservative[types.RHO]}
  cons_to_prim = {}
  for var_name in bc_dict:
    if types.is_conservative_name(var_name):
      if var_name in conservative:
        states[var_name] = conservative[var_name]
      else:
        raise ValueError(
            f'The variable "{var_name}" specified in the boundary condition is'
            ' not a valid conservative or primitive variable.'
        )
    else:  # var_name is a primitive variable name.
      cons_to_prim[types.primitive_to_conservative_name(var_name)] = (
          conservative[types.primitive_to_conservative_name(var_name)]
      )
  if len(cons_to_prim) > 1:
    prim_states = utils.conservative_to_primitive_variables(cons_to_prim)
    states.update(
        {
            var_name: val
            for var_name, val in prim_states.items()
            if var_name is not types.RHO
        }
    )

  return states


def update_boundary(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    states: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
) -> types.FlowFieldMap:
  """Updates the boundary conditions (BC).

  Args:
    replica_id: The global index of the current core.
    replicas: A 3D array stores the topology of the core distribution.
    states: A dictionary of flow field variables.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A flow field dictionary with boundary values updated.
  """
  return {
      key: exchange_halos(
          replica_id, replicas, val, cfg.bc[key], cfg.halo_width
      )
      for key, val in states.items()
  }
