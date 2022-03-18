import numpy as np
from nptyping import NDArray

Coordinate = NDArray[(3,), float]


def distance(a: Coordinate, b: Coordinate) -> float:
    """
    find the distance between 2 vectors
    """
    return np.linalg.norm(a - b) # better to use this than roll own version


def Gaussian(x:float, center:float, sigma:float) -> float:
    """
    get the value of the gaussian function at x 
    centered at `center` and with lengthscale `sigma`
    """
    d = distance(center, x)
    return np.exp(- 0.5 * d**2 / sigma**2)


 