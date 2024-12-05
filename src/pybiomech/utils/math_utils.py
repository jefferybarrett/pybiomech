import numpy as np

IDENTITY_3x3 = np.eye(3)
ZERO_3x3 = np.zeros((3,3))

def xprod_mat(r):
    x, y, z = r
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def rot(E):
    return np.block([[E, ZERO_3x3], [E, ZERO_3x3]])

def xlt(r):
    return np.block([[IDENTITY_3x3, ZERO_3x3], [-xprod_mat(r), IDENTITY_3x3]])




if __name__ == "__main__":
    r = np.array([1,2,3])
    print(xlt(r))
    print(xlt(-r).T)