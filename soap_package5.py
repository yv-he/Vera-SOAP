from itertools import product
from typing import Any, List

import numpy as np
import scipy
from ase import Atoms
from nptyping import NDArray
from scipy.special import sph_harm

from functions import Gaussian
from sphere_sampling import cartesian_product, sample_sphere_random

OneDArray = NDArray[(Any,), float]
TwoDArray = NDArray[(Any, Any), float]
Coordinates = NDArray[(Any, 3), float]
random = np.random.RandomState(seed=42).random


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


def g(rcut: float, n_max: int, r: float) -> OneDArray:
    overlap_matrix = overlaps(n_max)
    S_inv = np.linalg.inv(overlap_matrix)
    W = scipy.linalg.sqrtm(S_inv)

    ns = np.arange(n_max) + 1
    Phi = np.array([phi(rcut, r, n) for n in ns])

    return (W * Phi).sum(axis=1)


def Y(r: float, l_max: int) -> NDArray[OneDArray]:
    l = np.arange(0, l_max+1)  # l=0,1,...l_max
    Y = np.empty(len(l), dtype="object")
    for i in range(len(l)):
        y = []
        li = l[i]  # l=0,1,...l_max
        m = np.arange(-li, li+1)  # m = -l,-l+1,..0,..,l
        for j in range(2*i+1):
            y.append(sph_harm(m[j], li, r[2], r[1]))
        Y[i] = np.array(y)
    return Y


def soap_desc(
    atoms: Atoms, rcut: float, l_max: int, n_max: int, atom_sigma: float
) -> NDArray[OneDArray]:
    """
    generate soap descriptors for atoms, parameterised by the hypers
    """

    n_atom = len(atoms)
    alpha = 1/(atom_sigma**2)

    sphere_volume = 4/3*np.pi*rcut**3  # volume of the sphere

    r_cart, r_sph = sample_sphere_random(N=1000, r=rcut)

    Gn = np.empty(len(r_sph), dtype="object")  # raidal basis function
    for i in range(len(r_sph)):
        Gn[i] = g(rcut, n_max, r_sph[i][0])

    Ylm = np.empty(len(r_sph), dtype="object")  # spherical harmonics
    for i in range(len(r_sph)):
        Ylm[i] = Y(r_sph[i], l_max)

    P = np.empty(n_atom, dtype="object")
    for f in range(n_atom):

        # calculating the distance between central atom and all other atoms with pbc
        d = atoms.get_distances(np.arange(n_atom), f)
        D = atoms.get_distances(np.arange(n_atom), f,
                                vector=True)  # distance vectors
        Di = D[np.where(d < rcut)]  # distance vectors with a cutoff distance

        atom_neigh_den = np.empty(len(r_cart))  # atomic neighbour density
        for i in range(len(r_cart)):
            gaussian = []
            for j in range(len(Di)):
                gaussian.append(Gaussian(r_cart[i], Di[j], alpha))  # r-ri
            atom_neigh_den[i] = np.sum(gaussian)

        # extracting coefficients
        c_nlm = np.empty((n_max, l_max+1), dtype="object")
        for i in range(n_max):
            for j in range(l_max+1):
                c_n = []
                for k in range(2*j+1):
                    c_nl = []
                    for z in range(len(r_sph)):
                        c_nl.append(Gn[z][i]*Ylm[z][j][k]*atom_neigh_den[z])
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

        P[f] = p1

    return P
