import numpy as np
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


if __name__ == "__main__":
    test_sphere_sampling()
    test_conversions()
    print("Success - all tests pass")
