"""
Various utilities around voxel grids.
"""
import logging
import numpy as np
logger = logging.getLogger(__name__)

def convert_atom_to_voxel(molecule_xyz,
                          atom_index,
                          box_width,
                          voxel_width):
  """Converts atom coordinates to an i,j,k grid index.

  This function offsets the molecule atom coordinates by (box_width/2,
  box_width/2, box_width/2) and then divides by voxel_width to compute
  the voxel indices.

  Parameters:
  -----------
  molecule_xyz: np.ndarray
    Array with coordinates of all atoms in the molecule, shape
    (N, 3).
  atom_index: int
    Index of an atom in the molecule.
  box_width: float
    Size of the box in Angstroms.
  voxel_width: float
    Size of a voxel in Angstroms
  """

  indices = np.floor(
      (molecule_xyz[atom_index] + box_width / 2.0) / voxel_width).astype(int)
  if ((indices < 0) | (indices >= box_width / voxel_width)).any():
    logger.warning(
         'Coordinates are outside of the box (atom id = %s,'
         ' coords xyz = %s, coords in box = %s' %
         (atom_index, molecule_xyz[atom_index], indices))

  return ([indices])

def convert_atom_pair_to_voxel(molecule_xyz_tuple, atom_index_pair, box_width,
                               voxel_width):
  """Converts a pair of atoms to a list of i,j,k tuples."""

  indices_list = []
  indices_list.append(
      convert_atom_to_voxel(molecule_xyz_tuple[0], atom_index_pair[0],
                            box_width, voxel_width)[0])
  indices_list.append(
      convert_atom_to_voxel(molecule_xyz_tuple[1], atom_index_pair[1],
                            box_width, voxel_width)[0])
  return (indices_list)


def voxelize(get_voxels,
             voxels_per_edge,
             box_width,
             voxel_width,
             hash_function,
             coordinates,
             feature_dict=None,
             feature_list=None,
             nb_channel=16,
             dtype="np.int8"):
  """Helper function to voxelize inputs.

  This helper function helps convert a hash function which
  specifies spatial features of a molecular complex into a voxel
  tensor.

  Parameters
  ----------
  get_voxels: function
    Function that voxelizes inputs
  voxels_per_edge: int
    Number of voxels per one edge of box to voxelize.
  box_width: float, optional (default 16.0)
    Size of a box in which voxel features are calculated. Box
    is centered on a ligand centroid.
  voxel_width: float, optional (default 1.0)
    Size of a 3D voxel in a grid in Angstroms.
  hash_function: function
    Used to map feature choices to voxel channels.  
  coordinates: np.ndarray
    Contains the 3D coordinates of a molecular system.
  feature_dict: Dictionary
    Keys are atom indices.  
  feature_list: list
    List of available features. 
  nb_channel: int (Default 16)
    The number of feature channels computed per voxel. Should
    be a power of 2.
  dtype: type
    The dtype of the numpy ndarray created to hold features.

  Returns
  -------
  Tensor of shape (voxels_per_edge, voxels_per_edge,
  voxels_per_edge, nb_channel),
  """
  if dtype == "np.int8":
    feature_tensor = np.zeros(
        (voxels_per_edge, voxels_per_edge, voxels_per_edge,
         nb_channel),
        dtype=np.int8)
  else:
    feature_tensor = np.zeros(
        (voxels_per_edge, voxels_per_edge, voxels_per_edge,
         nb_channel),
        dtype=np.float16)
  if feature_dict is not None:
    for key, features in feature_dict.items():
      voxels = get_voxels(coordinates, key, box_width, voxel_width)
      for voxel in voxels:
        if ((voxel >= 0) & (voxel < voxels_per_edge)).all():
          if hash_function is not None:
            feature_tensor[voxel[0], voxel[1], voxel[2],
                           hash_function(features, nb_channel)] += 1.0
          else:
            feature_tensor[voxel[0], voxel[1], voxel[2], 0] += features
  elif feature_list is not None:
    for key in feature_list:
      voxels = get_voxels(coordinates, key, box_width, voxel_width)
      for voxel in voxels:
        if ((voxel >= 0) & (voxel < voxels_per_edge)).all():
          feature_tensor[voxel[0], voxel[1], voxel[2], 0] += 1.0

  return feature_tensor


