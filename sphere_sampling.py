"""
these functions assume the convention of spherical coordinates
commonly used in physics: (r, theta, phi)
where:
- 0 <=    r         = radius
- 0 <= theta < π    = polar angle (i.e. angle from positive z axis)
- 0 <=  phi  < 2π   = azimuthal angle (i.e. angle of rotation from the x axis)
"""

from typing import Any, Tuple

import numpy as np
from nptyping import NDArray


Coordinates = NDArray[(Any, 3), float]
random = np.random.RandomState(seed=42).random


def cart2sphr(cart: Coordinates) -> Coordinates:
    """
    convert cartesian (x, y, z) to spherical (r, theta, phi) coordinates
    """
    r = np.linalg.norm(cart, axis=1)

    x, y, z = cart.T

    # naively doing z / r gives division by zero when r = 0
    z_over_r = np.zeros_like(z)
    non_0_r = r != 0
    z_over_r[non_0_r] = z[non_0_r] / r[non_0_r]

    theta = np.arccos(z_over_r)
    phi = np.arctan2(y, x)

    return np.array([r, theta, phi]).T


def sphr2cart(sphr: Coordinates) -> Coordinates:
    """
    convert spherical (r, theta, phi) to cartesian (x, y, z) coordinates
    """

    r, theta, phi = sphr.T

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z]).T


def sample_unit_sphere_random(N: int) -> Tuple[Coordinates, Coordinates]:
    """
    generate N random points within the unit sphere
    """

    phi = random(N) * 2 * np.pi

    costheta = random(N) * 2 - 1
    theta = np.arccos(costheta)

    u = random(N)
    r = u ** (1/3)

    sphr = np.array([r, theta, phi]).T
    return sphr2cart(sphr), sphr


def sample_unit_sphere_uniform(N: int) -> Tuple[Coordinates, Coordinates]:
    """
    generate roughly N points with uniform spacing inside the unit sphere
    """

    v_sphere = 4 / 3 * np.pi
    v_cube = 2 ** 3
    total_N = N * v_cube / v_sphere

    grid_1d = np.linspace(-1, 1, num=int(total_N ** (1/3)))
    grid_3d = cartesian_product(grid_1d, grid_1d, grid_1d)

    cart = grid_3d[np.linalg.norm(grid_3d, axis=1) <= 1]
    return cart, cart2sphr(cart)


def sample_sphere_uniform(N: int, r: float) -> Tuple[Coordinates, Coordinates]:
    cart, sphr = sample_unit_sphere_uniform(N)
    cart *= r
    sphr[:, 0] *= r
    return cart, sphr


def sample_sphere_random(N: int, r: float) -> Tuple[Coordinates, Coordinates]:
    cart, sphr = sample_unit_sphere_random(N)
    cart *= r
    sphr[:, 0] *= r
    return cart, sphr


def cartesian_product(*arrays):
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))
