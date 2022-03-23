from typing import List
from ase import Atom
from ase.neighborlist import neighbor_list

from sphere_sampling import Coordinates


def local_atomic_environments(
    atoms: Atom, cutoff: float, include_central_atom: bool = True
) -> List[Coordinates]:
    """
    get a list of the vectors to each neighbour for each atom in atoms

    see https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html for explanation
    of arguments passed to neighbor_list, and why using this is preferable
    to implementing own methods (linear scaling!)

    # Example:
    >>> vectors_to_neighbours_for_ = neighbours_within_cutoff(atoms, cutoff=5)
    >>> atom_idx = 0
    >>> # get vectors to all atoms within cutoff of atom with index atom_id
    >>> vectors_to_neighbours_for_[atom_idx]
    [
        [x3, y3, z3],
        ...
        [x7, y7, z7],
    ]
    """

    atom_idxs, vectors = neighbor_list(
        "iD", atoms, cutoff=cutoff, self_interaction=include_central_atom
    )
    return [vectors[atom_idxs == i] for i in range(len(atoms))]
