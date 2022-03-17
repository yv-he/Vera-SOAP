import numpy as np

# converting between cartesian and spherical coord
# converting an array of coords


def cart2sph(xyz):
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


def sph2cart(rthetarho):
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

# converting single coord


def as_sph(xyz):
    x2y2 = xyz[0]**2 + xyz[1]**2
    r = np.sqrt(x2y2 + xyz[2]**2)
    theta = np.arctan2(np.sqrt(x2y2), xyz[2])
    rho = np.arctan2(xyz[1], xyz[0])
    return np.array([r, theta, rho])
    # definition for calculating norm of a vector

# definition for distance and distance vector


def distance(xyz, xyz1):
    # takes in two position vectors
    dx = xyz1[0] - xyz[0]
    dy = xyz1[1] - xyz[1]
    dz = xyz1[2] - xyz[2]
    distance = (dx**2+dy**2+dz**2)**0.5
    return distance


def dis_vector(xyz, xyz1):
    # takes in two position vectors
    x = xyz1[0] - xyz[0]
    y = xyz1[1] - xyz[1]
    z = xyz1[2] - xyz[2]
    return np.array([x, y, z])


# def distance(a, b):
#     return a - b


# Gaussian fucntion
def Gaussian(r, ri, alpha):
    gaussian = np.exp(-alpha*distance(ri, r)**2)
    return gaussian

# polynomial radial functions


def phi(rcut, r, n):
    N = np.sqrt(rcut**(2*n+5)/(2*n+5))  # normalisation
    phi = ((rcut-r)**(n+2))/N
    return phi
