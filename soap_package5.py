from typing import Any

import numpy as np
import scipy
from ase import Atoms
from ase.neighborlist import neighbor_list
from nptyping import NDArray
from scipy.special import sph_harm

from functions import Gaussian
from sphere_sampling import cartesian_product, sample_sphere_random

OneDArray = NDArray[(Any,), float]
TwoDArray = NDArray[(Any, Any), float]
Coordinates = NDArray[(Any, 3), float]
Coordinate = NDArray[(3), float]


N_SAMPLE = 1000  # number of points to sample for the atomic neighbour density


def phi(rcut: float, r: float, n: int) -> float:
    """
    value of radial basis function at r, paramterised by n and r_cut
    """

    N = np.sqrt(rcut**(2*n+5)/(2*n+5))
    phi = ((rcut-r)**(n+2))/N
    return phi


def overlaps(n_max: int) -> TwoDArray:
    """
    generate the matrix S of overlaps
    """
    ns = np.arange(n_max) + 1  # i.e. [1, 2, 3, ..., n_max]

    a, b = cartesian_product(ns, ns).T
    S = np.sqrt((2*a+5) * (2*b+5)) / (5+a+b)
    return S.reshape(n_max, n_max)


def overlaps_old(n_max: int) -> TwoDArray:
    n = np.arange(1, n_max+1)
    S = np.empty([n_max, n_max])
    for i in range(len(n)):
        a = n[i]  # alpha
        for j in range(len(n)):
            b = n[j]  # beta
            s = np.sqrt((2*a+5)*(2*b+5))/(5+a+b)  # overlap integral
            S[i][j] = s
    return S


def expand_in_rbfs(rcut: float, n_max: int, r: float) -> OneDArray:
    """
    expand the distance r in the radial basis functions 
    parameterised by rcut and n_max
    """

    # TODO this entire block is independent of r:
    # restructure your code so that this only needs to be calculated once
    ##################################
    overlap_matrix = overlaps(n_max)
    S_inv = np.linalg.inv(overlap_matrix)
    W = scipy.linalg.sqrtm(S_inv)
    ##################################

    ns = np.arange(n_max) + 1
    Phi = np.array([phi(rcut, r, n) for n in ns])

    # this can be generalised so that you pass in all the relevant
    # distances (rs) at once and then perform this vectorised calc.
    # this will make code much faster!
    return (W * Phi).sum(axis=1)


def expand_in_spherical_harmonics(
    sphr_coord: Coordinate, l_max: int
) -> NDArray[OneDArray]:
    """
    expand the (spherical) coordinate in spherical harmonics up to l=`l_max`
    """

    r, polar, azimuthal = sphr_coord

    l = np.arange(0, l_max+1)  # l=0,1,...l_max
    Y = np.empty(len(l), dtype="object")
    for i in range(len(l)):
        y = []
        li = l[i]  # l=0,1,...l_max
        m = np.arange(-li, li+1)  # m = -l,-l+1,..0,..,l
        for j in range(2*i+1):
            y.append(sph_harm(m[j], li, azimuthal, polar))
        Y[i] = np.array(y)

    # Y is of shape: [[1], [3], [5], [...], [2*l_max+1]]
    # In general want to avoid irregularly shaped arrs like this at all costs.
    return Y


def atomic_neighbour_density(
    atom_locations: Coordinates, sample_points: Coordinates, atom_sigma: float
) -> OneDArray:
    """
    calculate the atomic neighbour density at all `sample_points` 
    based on atoms at `atom_locations` and parameterised by `atom_sigma`
    """

    n_sample = len(sample_points)
    atom_neigh_den = np.empty(n_sample)
    for idx in range(n_sample):
        gaussians = [
            Gaussian(sample_points[idx], center=atom, sigma=atom_sigma)
            for atom in atom_locations
        ]
        atom_neigh_den[idx] = np.sum(gaussians)

    return atom_neigh_den


def soap_desc(
    atoms: Atoms, rcut: float, l_max: int, n_max: int, atom_sigma: float
) -> NDArray[OneDArray]:
    """
    generate soap descriptors for atoms, parameterised by the hypers
    """

    n_atom = len(atoms)

    sphere_volume = 4/3*np.pi*rcut**3  # volume of the sphere

    r_cart, r_sph = sample_sphere_random(N=N_SAMPLE, r=rcut)

    # positions of sampled points expanded in radial basis functions
    Gn = np.array([expand_in_rbfs(rcut, n_max, r) for r in r_sph[:, 0]])

    # positions of sampled points expanded as spherical harmonics
    Ylm = [
        expand_in_spherical_harmonics(sphr_coord, l_max)
        for sphr_coord in r_sph
    ]

    # see https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html
    _atoms, vectors = neighbor_list(
        "iD", atoms, cutoff=rcut, self_interaction=True
    )

    descriptor = np.empty(n_atom, dtype="object")
    for f in range(n_atom):
        neighbours = vectors[_atoms == f]

        # atomic neighbourhood density for atom f evaluated
        # at each point sampled in the sphere with radius rcut
        neigh_den = atomic_neighbour_density(neighbours, r_cart, atom_sigma)

        # extracting coefficients
        c_nlm = np.empty((n_max, l_max+1), dtype="object")
        for i in range(n_max):
            for j in range(l_max+1):
                c_n = []
                for k in range(2*j+1):
                    c_nl = []
                    for z in range(len(r_sph)):
                        c_nl.append(Gn[z][i]*Ylm[z][j][k]*neigh_den[z])
                    c_n.append((1/sphere_volume)*sum(c_nl))
                c_nlm[i][j] = np.array(c_n)

        c_nlm_conj = np.conj(c_nlm)  # complex conjugate of the coefficients

        p = []
        for i in range(n_max):
            cn1 = c_nlm_conj[i]
            for j in range(i, n_max):
                cn2 = c_nlm[j]
                for k in range(l_max+1):
                    pn = np.sum(cn1[k]*cn2[k]).real
                    p.append(2*np.pi*np.sqrt(8/(2*k+1))*pn)  # normalisation
        p = np.array(p)
        p1 = p / np.linalg.norm(p)

        descriptor[f] = p1

    return np.array(descriptor)
