from typing import Any, List
from ase import Atoms

import numpy as np
import scipy
from nptyping import NDArray
from scipy.special import sph_harm

from functions import Gaussian, cart2sph, distance

OneDArray = NDArray[(Any,), float]


def phi(rcut: float, r: float, n: int) -> float:
    """
    value of radial basis function at r, paramterised by n and r_cut
    """

    N = np.sqrt(rcut**(2*n+5)/(2*n+5))
    phi = ((rcut-r)**(n+2))/N
    return phi


def g(rcut: float, n_max: int, r: float) -> OneDArray:
    n = np.arange(1, n_max+1)
    # matrix containing overlap integrals of n and n'
    S = np.empty([n_max, n_max])
    for i in range(len(n)):
        a = n[i]  # alpha
        for j in range(len(n)):
            b = n[j]  # beta
            s = np.sqrt((2*a+5)*(2*b+5))/(5+a+b)  # overlap integral
            S[i][j] = s
    S_inv = np.linalg.inv(S)
    W = scipy.linalg.sqrtm(S_inv)  # W matrix
    Phi: List[float] = []
    for i in range(len(n)):
        Phi.append(phi(rcut, r, n[i]))
    g = np.empty(n_max)
    for i in range(n_max):
        gn = []
        for j in range(n_max):
            gn.append(W[j][i]*Phi[j])
        g[i] = np.sum(gn)
    return g

# spherical harmonics


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
    n_atom = len(atoms)
    a = 1/(atom_sigma**2)  # alpha
    # padding = atom_sigma*np.sqrt(-2*np.log(0.001)) #cutoff padding
    atom_pos = atoms.get_positions()  # postitions of atoms in the system
    atom_pos_sph = cart2sph(atom_pos)  # in spherical
    dr = 0.5
    n_r = int(2*rcut/dr)  # number of r points = n_r**3
    v = 4/3*np.pi*rcut**3  # volume of the sphere

    # sampling r points in a sphere
    r_cart = []
    coord = np.linspace(-rcut, rcut, n_r)
    for i in range(n_r):
        x = coord[i]
        for j in range(n_r):
            y = coord[j]
            for k in range(n_r):
                z = coord[k]
                if distance([x, y, z], [0, 0, 0]) <= rcut:
                    r_cart.append(np.array([x, y, z]))

    r_sph = cart2sph(r_cart)

    Gn = np.empty(len(r_sph), dtype="object")  # raidal basis function
    for i in range(len(r_sph)):
        Gn[i] = g(rcut, n_max, r_sph[i][0])
        # Gn = [[g1(r1),g2(r1)],[g1(r2),g2(r2)],[g1(r3),g2(r3)]...]

    Ylm = np.empty(len(r_sph), dtype="object")  # spherical harmonics
    for i in range(len(r_sph)):
        Ylm[i] = Y(r_sph[i], l_max)
        #Ylm = [[[Y00(r1)],[Y1-1(r1),Y10(r1),Y11(r1)],...],[[Y00(r2)],[Y1-1(r2),Y10(r2),Y11(r2)],...],...]

    P = np.empty(n_atom, dtype="object")
    for f in range(n_atom):
        # c_atom = atom_pos[f] #position of the central atom in cartesian
        #new_pos = atom_pos - c_atom
        # calculating the distance between central atom and all other atoms with pbc
        d = atoms.get_distances(np.arange(n_atom), f)
        D = atoms.get_distances(np.arange(n_atom), f,
                                vector=True)  # distance vectors
        Di = D[np.where(d < rcut)]  # distance vectors with a cutoff distance

        atom_neigh_den = np.empty(len(r_cart))  # atomic neighbour density
        for i in range(len(r_cart)):
            gaussian = []
            for j in range(len(Di)):
                gaussian.append(Gaussian(r_cart[i], Di[j], a))  # r-ri
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
                    c_n.append((1/v)*sum(c_nl))
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
