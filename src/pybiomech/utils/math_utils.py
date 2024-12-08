import numpy as np

IDENTITY_3x3 = np.eye(3)
ZERO_3x3 = np.zeros((3,3))
ZERO3 = np.zeros(3)

def mat4_from_components(E = IDENTITY_3x3, r = ZERO3):
    mat4 = np.eye(4)
    mat4[:3, :3] = E
    mat4[:3, 3] = r
    return mat4

def mat4_to_components(mat4):
     r = mat4[:3, 3]
     E = mat4[:3, :3]
     return E, r

def xprod_mat(r):
    x, y, z = r
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def rot(E):
    return np.block([[E, ZERO_3x3], [ZERO_3x3, E]])

def xlt(r):
    return np.block([[IDENTITY_3x3, ZERO_3x3], [-xprod_mat(r), IDENTITY_3x3]])




if __name__ == "__main__":
    r = np.array([1,2,3])
    print(xlt(r))
    print(xlt(-r).T)