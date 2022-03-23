import numpy as np
from ase.lattice.cubic import Diamond
from soap_package5 import atomic_neighbour_density, overlaps, overlaps_old, soap_desc
from sphere_sampling import cart2sphr, sample_unit_sphere_random, sample_unit_sphere_uniform, sphr2cart


def test_sphere_sampling():
    N = 400
    cart, sphr = sample_unit_sphere_random(N)
    assert len(cart) == len(sphr) == N, "Incorrect Number of Points Produced"
    assert (np.linalg.norm(cart, axis=1) < 1).all(), "Not All Points in Sphere"

    cart, sphr = sample_unit_sphere_uniform(N)


def test_conversions():
    zero = np.array([[0, 0, 0]])
    zero_sphr = cart2sphr(zero)
    zero_cart = sphr2cart(zero_sphr)
    assert (zero == zero_cart).all(), "origin not handled correctly"

    cart = np.random.randn(100, 3)
    sphr = cart2sphr(cart)
    cart_ = sphr2cart(sphr)
    assert cart.shape == sphr.shape == cart_.shape, "number of points not preserved"
    assert np.isclose(cart, cart_).all(), "Conversions Are Not Accurate"


def test_overlaps():
    n_max = 8
    S_old = overlaps_old(n_max)
    S_new = overlaps(n_max)

    assert (S_old == S_new).all(), "Incorrect implementation"


def test_neighbour_density():
    atoms = np.array([[0, 0, 0]])
    points = np.array([[0, 0, 0]])
    sigma = 0.5

    a_n_d = atomic_neighbour_density(atoms, points, sigma)
    assert a_n_d[0] == 1, "A.N.D. for single point and atom should be 1"


def test_soap():
    atoms = Diamond("C")
    atoms.set_pbc([False, False, False])
    soap_desc(atoms, rcut=3.7, l_max=1, n_max=4, atom_sigma=0.5)


if __name__ == "__main__":
    test_sphere_sampling()
    test_conversions()
    test_overlaps()
    test_soap()
    print("Success - all tests pass")
