from typing import List
import numpy as np
from nptyping import NDArray

Coordinate = NDArray[(3,), float]

def cart2sph(xyz: List[Coordinate]) -> List[Coordinate]:
    """
    converts a list of cartesian coordinates to spherical
    """
    sph = []
    for i in range(len(xyz)):
        coord = xyz[i]
        x2y2 = coord[0]**2 + coord[1]**2
        r = np.sqrt(x2y2 + coord[2]**2)  # radial distance r
        # azimuthal angle theta (defined from Z-axis down)
        theta = np.arctan2(np.sqrt(x2y2), coord[2])
        rho = np.arctan2(coord[1], coord[0])  # polar angle rho
        sph.append(np.array([r, theta, rho]))
    return sph


def sph2cart(rthetarho: List[Coordinate]) -> List[Coordinate]:
    """
    converts a list of spherical coordinates to spherical
    """
    cart = []
    for i in range(len(rthetarho)):
        coord = rthetarho[i]
        theta = coord[1]*np.pi/180  # azimuthal angle theta
        rho = coord[2]*np.pi/180  # polar angle rho
        x = coord[0]*np.sin(theta)*np.cos(rho)
        y = coord[0]*np.sin(theta)*np.sin(rho)
        z = coord[0]*np.cos(theta)
        cart.append(np.array([x, y, z]))
    return cart


def distance(a: Coordinate, b: Coordinate) -> float:
    """
    find the distance between 2 vectors
    """
    return np.linalg.norm(a - b) # better to use this than roll own version


def Gaussian(r:float, ri:float, alpha:float) -> float:
    """
    get the value of the gaussian function at r 
    with center ri and lengthscale alpha
    """
    return np.exp(-alpha*distance(ri, r)**2)


 