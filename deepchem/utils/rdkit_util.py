"""
RDKit Utilities.

This file contains utilities that compute useful properties of
molecules. Some of these are simple cleanup utilities, and
others are more sophisticated functions that detect chemical
properties of molecules.
"""

import os
import logging
import numpy as np
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO
from copy import deepcopy
from collections import Counter
from scipy.spatial.distance import cdist
from deepchem.utils.geometry_utils import angle_between
from deepchem.utils.geometry_utils import generate_random_rotation_matrix

logger = logging.getLogger(__name__)

class MoleculeLoadException(Exception):

  def __init__(self, *args, **kwargs):
    Exception.__init__(*args, **kwargs)


def compute_pairwise_distances(first_xyz, second_xyz):
  """Computes pairwise distances between two molecules.

  Takes an input (m, 3) and (n, 3) numpy arrays of 3D coords of
  two molecules respectively, and outputs an m x n numpy
  array of pairwise distances in Angstroms between the first and
  second molecule. entry (i,j) is dist between the i"th 
  atom of first molecule and the j"th atom of second molecule.

  Parameters
  ----------
  first_xyz: np.ndarray
    Of shape (m, 3)
  seocnd_xyz: np.ndarray
    Of shape (n, 3)

  Returns
  -------
  np.ndarray of shape (m, n)
  """

  pairwise_distances = cdist(first_xyz, second_xyz,
                             metric='euclidean')
  return (pairwise_distances)

def get_xyz_from_mol(mol):
  """
  returns an m x 3 np array of 3d coords
  of given rdkit molecule
  """
  xyz = np.zeros((mol.GetNumAtoms(), 3))
  conf = mol.GetConformer()
  for i in range(conf.GetNumAtoms()):
    position = conf.GetAtomPosition(i)
    xyz[i, 0] = position.x
    xyz[i, 1] = position.y
    xyz[i, 2] = position.z
  return (xyz)


def add_hydrogens_to_mol(mol):
  """
  Add hydrogens to a molecule object

  TODO(rbharath) see if there are more flags to add here for default

  Parameters
  ----------
  mol: Rdkit Mol
    Molecule to hydrogenate

  Returns
  -------
  Rdkit Mol
  """
  molecule_file = None
  try:
    from rdkit import Chem
    pdbblock = Chem.MolToPDBBlock(mol)
    pdb_stringio = StringIO()
    pdb_stringio.write(pdbblock)
    pdb_stringio.seek(0)
    import pdbfixer
    fixer = pdbfixer.PDBFixer(pdbfile=pdb_stringio)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    hydrogenated_io = StringIO()
    import simtk
    simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions,
                                       hydrogenated_io)
    hydrogenated_io.seek(0)
    return Chem.MolFromPDBBlock(
        hydrogenated_io.read(), sanitize=False, removeHs=False)
  except ValueError as e:
    logging.warning("Unable to add hydrogens %s", e)
    raise MoleculeLoadException(e)
  finally:
    try:
      os.remove(molecule_file)
    except (OSError, TypeError):
      pass


def compute_charges(mol):
  """Attempt to compute Gasteiger Charges on Mol

  This also has the side effect of calculating charges on mol.
  The mol passed into this function has to already have been sanitized

  Params
  ------
  mol: rdkit molecule

  Returns
  -------
  molecule with charges
  """
  from rdkit.Chem import AllChem
  try:
    AllChem.ComputeGasteigerCharges(mol)
  except Exception as e:
    logging.exception("Unable to compute charges for mol")
    raise MoleculeLoadException(e)
  return mol


def load_molecule(molecule_file,
                  add_hydrogens=True,
                  calc_charges=True,
                  sanitize=True):
  """Converts molecule file to (xyz-coords, obmol object)

  Given molecule_file, returns a tuple of xyz coords of molecule
  and an rdkit object representing that molecule

  Parameters
  ----------
  molecule_file: str
    filename for molecule
  add_hydrogens: bool, optional
    If true, add hydrogens via pdbfixer
  calc_charges: bool, optional
    If true, add charges via rdkit
  sanitize: bool, optional
    If true, sanitize molecules via rdkit

  Returns
  -------
  Tuple (xyz, mol)
  """
  from rdkit import Chem
  if ".mol2" in molecule_file:
    my_mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
  elif ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
    my_mol = suppl[0]
  elif ".pdbqt" in molecule_file:
    pdb_block = pdbqt_to_pdb(molecule_file)
    my_mol = Chem.MolFromPDBBlock(
        str(pdb_block), sanitize=False, removeHs=False)
  elif ".pdb" in molecule_file:
    my_mol = Chem.MolFromPDBFile(
        str(molecule_file), sanitize=False, removeHs=False)
  else:
    raise ValueError("Unrecognized file type")

  if my_mol is None:
    raise ValueError("Unable to read non None Molecule Object")

  if add_hydrogens or calc_charges:
    my_mol = add_hydrogens_to_mol(my_mol)
  if sanitize:
    Chem.SanitizeMol(my_mol)
  if calc_charges:
    compute_charges(my_mol)

  xyz = get_xyz_from_mol(my_mol)

  return xyz, my_mol

def convert_protein_to_pdbqt(mol, outfile):
  """Convert a protein PDB file into a pdbqt file.

  Parameters
  ----------
  mol: rdkit Mol
    Protein molecule
  outfile: str
    filename which already has a valid pdb representation of mol
  """
  lines = [x.strip() for x in open(outfile).readlines()]
  out_lines = []
  for line in lines:
    if "ROOT" in line or "ENDROOT" in line or "TORSDOF" in line:
      out_lines.append("%s\n" % line)
      continue
    if not line.startswith("ATOM"):
      continue
    line = line[:66]
    atom_index = int(line.split()[1])
    atom = mol.GetAtoms()[atom_index - 1]
    line = "%s    +0.000 %s\n" % (line, atom.GetSymbol().ljust(2))
    out_lines.append(line)
  with open(outfile, 'w') as fout:
    for line in out_lines:
      fout.write(line)


def convert_ligand_to_pdbqt(mol, outfile):
  """Convert a pdb ligand into a pdbqt ligand

  Parameters
  ----------
  mol: rdkit Mol
    Ligand molecule
  outfile: str
    filename which already has a valid pdb representation of mol
  """
  PdbqtLigandWriter(mol, outfile).convert()


def write_molecule(mol, outfile, is_protein=False):
  """Write molecule to a file

  Parameters
  ----------
  mol: rdkit Mol
    Molecule to write
  outfile: str
    Filename to write mol to
  is_protein: bool, optional
    Is this molecule a protein?
  """
  from rdkit import Chem
  if ".pdbqt" in outfile:
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
    if is_protein:
      convert_protein_to_pdbqt(mol, outfile)
    else:
      convert_ligand_to_pdbqt(mol, outfile)
  elif ".pdb" in outfile:
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
  elif ".sdf" in outfile:
    writer = Chem.SDWriter(outfile)
    writer.write(mol)
    writer.close()
  else:
    raise ValueError("Unsupported Format")


def pdbqt_to_pdb(filename):
  pdbqt_data = open(filename).readlines()
  pdb_block = ""
  for line in pdbqt_data:
    pdb_block += "%s\n" % line[:66]
  return pdb_block


def merge_molecules_xyz(first_xyz, second_xyz):
  """Merges coordinates of two molecules.
  """
  return np.array(np.vstack(np.vstack((first_xyz, second_xyz))))


def merge_molecules(first, second):
  """Helper method to merge two molecules."""
  from rdkit.Chem import rdmolops
  return rdmolops.CombineMols(first, second)


class PdbqtLigandWriter(object):
  """
  Create a torsion tree and write to pdbqt file
  """

  def __init__(self, mol, outfile):
    """
    Parameters
    ----------
    mol: rdkit Mol
      The molecule whose value is stored in pdb format in outfile
    outfile: str
      Filename for a valid pdb file with the extention .pdbqt
    """
    self.mol = mol
    self.outfile = outfile

  def convert(self):
    """
    The single public function of this class.
    It converts a molecule and a pdb file into a pdbqt file stored in outfile
    """
    import networkx as nx
    self._create_pdb_map()
    self._mol_to_graph()
    self._get_rotatable_bonds()

    for bond in self.rotatable_bonds:
      self.graph.remove_edge(bond[0], bond[1])
    self.components = [x for x in nx.connected_components(self.graph)]
    self._create_component_map(self.components)

    self.used_partitions = set()
    self.lines = []
    root = max(enumerate(self.components), key=lambda x: len(x[1]))[0]
    self.lines.append("ROOT\n")
    for atom in self.components[root]:
      self.lines.append(self._writer_line_for_atom(atom))
    self.lines.append("ENDROOT\n")
    self.used_partitions.add(root)
    for bond in self.rotatable_bonds:
      valid, next_partition = self._valid_bond(bond, root)
      if not valid:
        continue
      self._dfs(next_partition, bond)
    self.lines.append("TORSDOF %s" % len(self.rotatable_bonds))
    with open(self.outfile, 'w') as fout:
      for line in self.lines:
        fout.write(line)

  def _dfs(self, current_partition, bond):
    """
    This function does a depth first search throught he torsion tree
    :param current_partition: The current partition to expand
    :param bond: the bond which goes from the previous partition into this partition
    """
    if self._get_component_for_atom(bond[1]) != current_partition:
      bond = (bond[1], bond[0])
    self.used_partitions.add(self._get_component_for_atom(bond[0]))
    self.used_partitions.add(self._get_component_for_atom(bond[1]))
    self.lines.append("BRANCH %4s %4s\n" % (bond[0] + 1, bond[1] + 1))
    for atom in self.components[current_partition]:
      self.lines.append(self._writer_line_for_atom(atom))
    for b in self.rotatable_bonds:
      valid, next_partition = self._valid_bond(b, current_partition)
      if not valid:
        continue
      self._dfs(next_partition, b)
    self.lines.append("ENDBRANCH %4s %4s\n" % (bond[0] + 1, bond[1] + 1))

  def _get_component_for_atom(self, atom_number):
    """
    :param atom_number: the atom number to check for component_id
    :return: the component_id that atom_number is part of
    """
    return self.comp_map[atom_number]

  def _valid_bond(self, bond, current_partition):
    """
    used to check if a bond goes from the current partition into a partition
    that is not yet explored
    :param bond: the bond to check if it goes to an unexplored partition
    :param current_partition: the current partition of the DFS
    :return: is_valid, next_partition
    """
    part1 = self.comp_map[bond[0]]
    part2 = self.comp_map[bond[1]]
    if part1 != current_partition and part2 != current_partition:
      return False, 0
    if part1 == current_partition:
      next_partition = part2
    else:
      next_partition = part1
    return not next_partition in self.used_partitions, next_partition

  def _writer_line_for_atom(self, atom_number):
    """

    :param atom_number:
    :return:
    """
    return self.pdb_map[atom_number]

  def _create_component_map(self, components):
    """Creates a Map From atom_idx to disconnected_component_id

    Sets self.comp_map to the computed compnent map.

    Parameters
    ----------
    components: list
      List of connected components
    """
    comp_map = {}
    for i in range(self.mol.GetNumAtoms()):
      for j in range(len(components)):
        if i in components[j]:
          comp_map[i] = j
          break
    self.comp_map = comp_map

  def _create_pdb_map(self):
    """Create self.pdb_map.

    This is a map from rdkit atom number to its line in the pdb
    file. We also add the two additional columns required for
    pdbqt (charge, symbol)

    note rdkit atoms are 0 indexes and pdb files are 1 indexed
    """
    lines = [x.strip() for x in open(self.outfile).readlines()]
    lines = filter(lambda x: x.startswith("HETATM") or x.startswith("ATOM"),
                   lines)
    lines = [x[:66] for x in lines]
    pdb_map = {}
    for line in lines:
      my_values = line.split()
      atom_number = int(my_values[1])
      atom_symbol = my_values[2]
      atom_symbol = ''.join([i for i in atom_symbol if not i.isdigit()])
      line = line.replace("HETATM", "ATOM  ")
      line = "%s    +0.000 %s\n" % (line, atom_symbol.ljust(2))
      pdb_map[atom_number - 1] = line
    self.pdb_map = pdb_map

  def _mol_to_graph(self):
    """
    Convert self.mol into a graph representation
    atoms are nodes, and bonds are vertices
    store as self.graph
    """
    import networkx as nx
    G = nx.Graph()
    num_atoms = self.mol.GetNumAtoms()
    G.add_nodes_from(range(num_atoms))
    for i in range(self.mol.GetNumBonds()):
      from_idx = self.mol.GetBonds()[i].GetBeginAtomIdx()
      to_idx = self.mol.GetBonds()[i].GetEndAtomIdx()
      G.add_edge(from_idx, to_idx)
    self.graph = G

  def _get_rotatable_bonds(self):
    """
    https://github.com/rdkit/rdkit/blob/f4529c910e546af590c56eba01f96e9015c269a6/Code/GraphMol/Descriptors/Lipinski.cpp#L107
    Taken from rdkit source to find which bonds are rotatable
    store rotatable bonds in (from_atom, to_atom)
    """
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    pattern = Chem.MolFromSmarts(
        "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("
        "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="
        "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#"
        "*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])"
        "[CH3])]")
    rdmolops.FastFindRings(self.mol)
    self.rotatable_bonds = self.mol.GetSubstructMatches(pattern)

def compute_centroid(coordinates):
  """Compute the x,y,z centroid of provided coordinates

  coordinates: np.ndarray
    Shape (N, 3), where N is number atoms.
  """
  centroid = np.mean(coordinates, axis=0)
  return (centroid)

def subtract_centroid(xyz, centroid):
  """Subtracts centroid from each coordinate.

  Subtracts the centroid, a numpy array of dim 3, from all coordinates of all
  atoms in the molecule
  """
  xyz -= np.transpose(centroid)
  return (xyz)

def compute_ring_center(mol, ring_indices):
  """Computes 3D coordinates of a center of a given ring.

  Parameters:
  -----------
  mol: rdkit.rdchem.Mol
    Molecule containing a ring
  ring_indices: array-like
    Indices of atoms forming a ring

  Returns:
  --------
    ring_centroid: np.ndarray
      Position of a ring center
  """
  conformer = mol.GetConformer()
  ring_xyz = np.zeros((len(ring_indices), 3))
  for i, atom_idx in enumerate(ring_indices):
    atom_position = conformer.GetAtomPosition(atom_idx)
    ring_xyz[i] = np.array(atom_position)
  ring_centroid = compute_centroid(ring_xyz)
  return ring_centroid
 
def compute_ring_normal(mol, ring_indices):
  """Computes normal to a plane determined by a given ring.

  Parameters:
  -----------
  mol: rdkit.rdchem.Mol
    Molecule containing a ring
  ring_indices: array-like
    Indices of atoms forming a ring

  Returns:
  --------
  normal: np.ndarray
    Normal vector
  """
  conformer = mol.GetConformer()
  points = np.zeros((3, 3))
  for i, atom_idx in enumerate(ring_indices[:3]):
    atom_position = conformer.GetAtomPosition(atom_idx)
    points[i] = np.array(atom_position)

  v1 = points[1] - points[0]
  v2 = points[2] - points[0]
  normal = np.cross(v1, v2)
  return normal

def rotate_molecules(mol_coordinates_list):
  """Rotates provided molecular coordinates.

  Pseudocode:
  1. Generate random rotation matrix. This matrix applies a
     random transformation to any 3-vector such that, were the
     random transformation repeatedly applied, it would randomly
     sample along the surface of a sphere with radius equal to
     the norm of the given 3-vector cf.
     generate_random_rotation_matrix() for details
  2. Apply R to all atomic coordinates.
  3. Return rotated molecule

  Parameters
  ----------
  mol_coordinates_list: list
    Elements of list must be (N_atoms, 3) shaped arrays
  """
  R = generate_random_rotation_matrix()
  rotated_coordinates_list = []

  for mol_coordinates in mol_coordinates_list:
    coordinates = deepcopy(mol_coordinates)
    rotated_coordinates = np.transpose(np.dot(R, np.transpose(coordinates)))
    rotated_coordinates_list.append(rotated_coordinates)

  return (rotated_coordinates_list)

def get_partial_charge(atom):
  """Get partial charge of a given atom (rdkit Atom object)"""
  try:
    value = atom.GetProp(str("_GasteigerCharge"))
    if value == '-nan':
      return 0
    return float(value)
  except KeyError:
    return 0

def is_salt_bridge(atom_i, atom_j):
  """Check if two atoms have correct charges to form a salt bridge"""
  if np.abs(2.0 - np.abs(
      get_partial_charge(atom_i) - get_partial_charge(atom_j))) < 0.01:
    return True
  return False

def is_hydrogen_bond(protein_xyz,
                     protein,
                     ligand_xyz,
                     ligand,
                     contact,
                     hbond_angle_cutoff):
  """
  Determine if a pair of atoms (contact = tuple of protein_atom_index, ligand_atom_index)
  between protein and ligand represents a hydrogen bond. Returns a boolean result.
  """
  # TODO(rbharath)
  return False

def compute_salt_bridges(protein,
                         ligand,
                         pairwise_distances,
                         cutoff=5.0):
  """Find salt bridge contacts between protein and ligand.

  Parameters:
  -----------
  protein: rdkit.rdchem.Mol
    Interacting molecules
  ligand: rdkit.rdchem.Mol
    Interacting molecules
  pairwise_distances: np.ndarray
    Array of pairwise protein-ligand distances (Angstroms)
  cutoff: float
    Cutoff distance for contact consideration

  Returns:
  --------
  salt_bridge_contacts: list of tuples
    List of contacts. Tuple (i, j) indicates that atom i from protein
    interacts with atom j from ligand.
  """

  salt_bridge_contacts = []
  contacts = np.nonzero(pairwise_distances < cutoff)
  contacts = zip(contacts[0], contacts[1])
  for contact in contacts:
    protein_atom = protein.GetAtoms()[int(contact[0])]
    ligand_atom = ligand.GetAtoms()[int(contact[1])]
    if is_salt_bridge(protein_atom, ligand_atom):
      salt_bridge_contacts.append(contact)
  return salt_bridge_contacts

def compute_cation_pi(mol1, mol2, charge_tolerance=0.01, **kwargs):
  """Finds aromatic rings in mo1 and cations in mol2 that interact with each other.

  Parameters:
  -----------
  mol1: rdkit.rdchem.Mol
    Molecule to look for interacting rings
  mol2: rdkit.rdchem.Mol
    Molecule to look for interacting cations
  charge_tolerance: float
    Atom is considered a cation if its formal charge is greater
    than 1 - charge_tolerance
  **kwargs:
    Arguments that are passed to is_cation_pi function

  Returns:
  --------
  mol1_pi: dict
    Dictionary that maps atom indices (from mol1) to the number of cations
    (in mol2) they interact with
  mol2_cation: dict
    Dictionary that maps atom indices (from mol2) to the number of aromatic
    atoms (in mol1) they interact with
  """
  mol1_pi = Counter()
  mol2_cation = Counter()
  conformer = mol2.GetConformer()

  aromatic_atoms = set(atom.GetIdx() for atom in mol1.GetAromaticAtoms())
  from rdkit import Chem
  rings = [list(r) for r in Chem.GetSymmSSSR(mol1)]

  for ring in rings:
    # if ring from mol1 is aromatic
    if set(ring).issubset(aromatic_atoms):
      ring_center = compute_ring_center(mol1, ring)
      ring_normal = compute_ring_normal(mol1, ring)

      for atom in mol2.GetAtoms():
        # ...and atom from mol2 is a cation
        if atom.GetFormalCharge() > 1.0 - charge_tolerance:
          cation_position = np.array(conformer.GetAtomPosition(atom.GetIdx()))
          # if angle and distance are correct
          if is_cation_pi(cation_position, ring_center, ring_normal, **kwargs):
            # count atoms forming a contact
            mol1_pi.update(ring)
            mol2_cation.update([atom.GetIndex()])
  return mol1_pi, mol2_cation

def is_cation_pi(cation_position,
                 ring_center,
                 ring_normal,
                 dist_cutoff=6.5,
                 angle_cutoff=30.0):
  """Check if a cation and an aromatic ring form contact.

  Parameters:
  -----------
    ring_center: np.ndarray
      Positions of ring center. Can be computed with the compute_ring_center
      function.
    ring_normal: np.ndarray
      Normal of ring. Can be computed with the compute_ring_normal function.
    dist_cutoff: float
      Distance cutoff. Max allowed distance between ring center and cation
      (in Angstroms).
    angle_cutoff: float
      Angle cutoff. Max allowed deviation from the ideal (0deg) angle between
      ring normal and vector pointing from ring center to cation (in degrees).
  """
  cation_to_ring_vec = cation_position - ring_center
  dist = np.linalg.norm(cation_to_ring_vec)
  angle = angle_between(cation_to_ring_vec, ring_normal) * 180. / np.pi
  if ((angle < angle_cutoff or angle > 180.0 - angle_cutoff) and
      (dist < dist_cutoff)):
    return True
  return False

def compute_pi_stack(protein,
                     ligand,
                     pairwise_distances=None,
                     dist_cutoff=4.4,
                     angle_cutoff=30.):
  """Find aromatic rings in protein and ligand that form pi-pi contacts.
  For each atom in the contact, count number of atoms in the other molecule
  that form this contact.

  Pseudocode:

  for each aromatic ring in protein:
    for each aromatic ring in ligand:
      compute distance between centers
      compute angle between normals
      if it counts as parallel pi-pi:
        count interacting atoms
      if it counts as pi-T:
        count interacting atoms

  Parameters:
  -----------
    protein, ligand: rdkit.rdchem.Mol
      Two interacting molecules.
    pairwise_distances: np.ndarray (optional)
      Array of pairwise protein-ligand distances (Angstroms)
    dist_cutoff: float
      Distance cutoff. Max allowed distance between the ring center (Angstroms).
    angle_cutoff: float
      Angle cutoff. Max allowed deviation from the ideal angle between rings.

  Returns:
  --------
    protein_pi_t, protein_pi_parallel, ligand_pi_t, ligand_pi_parallel: dict
      Dictionaries mapping atom indices to number of atoms they interact with.
      Separate dictionary is created for each type of pi stacking (parallel and
      T-shaped) and each molecule (protein and ligand).
  """

  protein_pi_parallel = Counter()
  protein_pi_t = Counter()
  ligand_pi_parallel = Counter()
  ligand_pi_t = Counter()

  protein_aromatic_rings = []
  ligand_aromatic_rings = []
  from rdkit import Chem
  for mol, ring_list in ((protein, protein_aromatic_rings),
                         (ligand, ligand_aromatic_rings)):
    aromatic_atoms = {atom.GetIdx() for atom in mol.GetAromaticAtoms()}
    for ring in Chem.GetSymmSSSR(mol):
      # if ring is aromatic
      if set(ring).issubset(aromatic_atoms):
        # save its indices, center, and normal
        ring_center = compute_ring_center(mol, ring)
        ring_normal = compute_ring_normal(mol, ring)
        ring_list.append((ring, ring_center, ring_normal))

  # remember protein-ligand pairs we already counted
  counted_pairs_parallel = set()
  counted_pairs_t = set()
  for prot_ring, prot_ring_center, prot_ring_normal in protein_aromatic_rings:
    for lig_ring, lig_ring_center, lig_ring_normal in ligand_aromatic_rings:
      if is_pi_parallel(
          prot_ring_center,
          prot_ring_normal,
          lig_ring_center,
          lig_ring_normal,
          angle_cutoff=angle_cutoff,
          dist_cutoff=dist_cutoff):
        prot_to_update = set()
        lig_to_update = set()
        for prot_atom_idx in prot_ring:
          for lig_atom_idx in lig_ring:
            if (prot_atom_idx, lig_atom_idx) not in counted_pairs_parallel:
              # if this pair is new, count atoms forming a contact
              prot_to_update.add(prot_atom_idx)
              lig_to_update.add(lig_atom_idx)
              counted_pairs_parallel.add((prot_atom_idx, lig_atom_idx))

        protein_pi_parallel.update(prot_to_update)
        ligand_pi_parallel.update(lig_to_update)

      if is_pi_t(
          prot_ring_center,
          prot_ring_normal,
          lig_ring_center,
          lig_ring_normal,
          angle_cutoff=angle_cutoff,
          dist_cutoff=dist_cutoff):
        prot_to_update = set()
        lig_to_update = set()
        for prot_atom_idx in prot_ring:
          for lig_atom_idx in lig_ring:
            if (prot_atom_idx, lig_atom_idx) not in counted_pairs_t:
              # if this pair is new, count atoms forming a contact
              prot_to_update.add(prot_atom_idx)
              lig_to_update.add(lig_atom_idx)
              counted_pairs_t.add((prot_atom_idx, lig_atom_idx))

        protein_pi_t.update(prot_to_update)
        ligand_pi_t.update(lig_to_update)

  return (protein_pi_t, protein_pi_parallel, ligand_pi_t, ligand_pi_parallel)

def is_pi_t(ring1_center,
            ring1_normal,
            ring2_center,
            ring2_normal,
            dist_cutoff=5.5,
            angle_cutoff=30.0):
  """Check if two aromatic rings form a T-shaped pi-pi contact.

  Parameters:
  -----------
  ring1_center, ring2_center: np.ndarray
    Positions of centers of the two rings. Can be computed with the
    compute_ring_center function.
  ring1_normal, ring2_normal: np.ndarray
    Normals of the two rings. Can be computed with the compute_ring_normal
    function.
  dist_cutoff: float
    Distance cutoff. Max allowed distance between the ring center (Angstroms).
  angle_cutoff: float
    Angle cutoff. Max allowed deviation from the ideal (90deg) angle between
    the rings (in degrees).
  """
  dist = np.linalg.norm(ring1_center - ring2_center)
  angle = angle_between(ring1_normal, ring2_normal) * 180 / np.pi
  if ((90.0 - angle_cutoff < angle < 90.0 + angle_cutoff) and
      dist < dist_cutoff):
    return True
  return False

def is_pi_parallel(ring1_center,
                   ring1_normal,
                   ring2_center,
                   ring2_normal,
                   dist_cutoff=8.0,
                   angle_cutoff=30.0):
  """Check if two aromatic rings form a parallel pi-pi contact.

  Parameters:
  -----------
    ring1_center, ring2_center: np.ndarray
      Positions of centers of the two rings. Can be computed with the
      compute_ring_center function.
    ring1_normal, ring2_normal: np.ndarray
      Normals of the two rings. Can be computed with the compute_ring_normal
      function.
    dist_cutoff: float
      Distance cutoff. Max allowed distance between the ring center (Angstroms).
    angle_cutoff: float
      Angle cutoff. Max allowed deviation from the ideal (0deg) angle between
      the rings (in degrees).
  """

  dist = np.linalg.norm(ring1_center - ring2_center)
  angle = angle_between(ring1_normal, ring2_normal) * 180 / np.pi
  if ((angle < angle_cutoff or angle > 180.0 - angle_cutoff) and
      dist < dist_cutoff):
    return True
  return False

def compute_binding_pocket_cation_pi(protein, ligand, **kwargs):
  """Finds cation-pi interactions between protein and ligand.

  Parameters:
  -----------
    protein, ligand: rdkit.rdchem.Mol
      Interacting molecules
    **kwargs:
      Arguments that are passed to compute_cation_pi function

  Returns:
  --------
    protein_cation_pi, ligand_cation_pi: dict
      Dictionaries that maps atom indices to the number of cations/aromatic
      atoms they interact with
  """
  # find interacting rings from protein and cations from ligand
  protein_pi, ligand_cation = compute_cation_pi(protein, ligand, **kwargs)
  # find interacting cations from protein and rings from ligand
  ligand_pi, protein_cation = compute_cation_pi(ligand, protein, **kwargs)

  # merge counters
  protein_cation_pi = Counter()
  protein_cation_pi.update(protein_pi)
  protein_cation_pi.update(protein_cation)

  ligand_cation_pi = Counter()
  ligand_cation_pi.update(ligand_pi)
  ligand_cation_pi.update(ligand_cation)

  return protein_cation_pi, ligand_cation_pi

def compute_all_ecfp(mol, indices=None, degree=2):
  """Obtain molecular fragment for all atoms emanating outward to given degree.

  For each fragment, compute SMILES string (for now) and hash to
  an int. Return a dictionary mapping atom index to hashed
  SMILES.
  """

  ecfp_dict = {}
  from rdkit import Chem
  for i in range(mol.GetNumAtoms()):
    if indices is not None and i not in indices:
      continue
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, degree, i, useHs=True)
    submol = Chem.PathToSubmol(mol, env)
    smile = Chem.MolToSmiles(submol)
    ecfp_dict[i] = "%s,%s" % (mol.GetAtoms()[i].GetAtomicNum(), smile)

  return ecfp_dict
