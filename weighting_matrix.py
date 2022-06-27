from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Any
from nptyping import NDArray
import scipy
from sphere_sampling import cartesian_product
from scipy import linalg

OneDArray = NDArray[(Any,), float]
TwoDArray = NDArray[(Any, Any), float]
Coordinate = NDArray[(3), float]

def overlaps(n_max: int) -> TwoDArray:
    """
    generate the matrix S of overlaps
    """
    ns = np.arange(n_max) + 1  # i.e. [1, 2, 3, ..., n_max]

    a, b = cartesian_product(ns, ns).T
    S = np.sqrt((2*a+5) * (2*b+5)) / (5+a+b)
    return S.reshape(n_max, n_max)

def wmatrix(n_max:int) -> TwoDArray:
    """
    generate the weighting matrix W
    """
    S = overlaps(n_max=n_max)
    S_inv = np.linalg.inv(S)
    return scipy.linalg.sqrtm(S_inv)

W = wmatrix(12)
np.save("wmatrix", W)