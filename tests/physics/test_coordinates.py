import numpy as np
from pybiomech.physics.coordinates import *
from scipy.spatial.transform import Rotation

def test_coord_transforms():
    for _ in range(100):
        alpha, beta, gamma = [float(x) for x in 2*np.pi * np.random.rand(3)]
        R = Rotation.from_euler('xyz', [alpha, beta, gamma]).as_matrix()
        dr = 100 * (np.random.rand(3) - 0.5)
        
        # transformation
        T_ab = CoordinateTransformation(rotation=R, translation=dr)

        motion = SpatialMotion(100*np.random.rand(6))
        force = SpatialForce(100*np.random.rand(6))
        
        # this is a pretty strong force. The power doesn't depend on the coordinate
        # system we're analyzing in, so these should always be the same
        assert np.isclose(motion.dot(force), (T_ab @ motion).dot(T_ab @ force))
        
    
def test_coordinate_transform():
    E = Rotation.from_euler('xyz', [45, 23, 10], degrees = True).as_matrix()
    r = 100 * (np.random.rand(3)-0.5)
    
    T = CoordinateTransformation(rotation = E, translation = r)
    
    v = 100 * (np.random.rand(3)-0.5)
    vh = np.hstack([v, [1]]) # homogeneous form
    res = (T @ vh)[0:3]
    assert np.all(np.isclose(T @ v, E @ v + r))
    assert np.all(np.isclose(res, E @ v + r))
    
    
def test_coordinate_inverse():
    E = Rotation.from_euler('xyz', [45, 23, 10], degrees = True).as_matrix()
    r = 100 * (np.random.rand(3)-0.5)
    T = CoordinateTransformation(rotation = E, translation = r)
    Tinv = T.inv()
    
    v = 100 * (np.random.rand(3)-0.5)
    assert np.all(np.isclose(T @ Tinv @ v, v))
    assert np.all(np.isclose(Tinv @ T @ v, v))


def test_frame_construction():
    rotation = Rotation.from_euler('xyz', [25, 32, 35], degrees=True).as_matrix()
    translation = np.array([1.0, 2.0, -2.3])
    G = Frame.from_orign_orientation(origin = translation, orientation = rotation)
    A = Frame.orient_wrt(G, rotation, translation)

    mat4 = mat4_from_translation_rotation(translation, rotation)
    T = CoordinateTransformation.between_frames(G, A)
    assert np.all(np.isclose(T.mat, mat4))

def test_frame_construction_id():
    rotation = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    offset = np.random.rand(3)*100
    G = Frame.from_orign_orientation(origin = offset, orientation = rotation)
    #A = Frame.orient_wrt(G, np.eye(3), np.zeros(3))
    A = Frame.from_mat4(G.as_mat4())

    T = CoordinateTransformation.between_frames(G, A)
    assert np.all(np.isclose(T.mat, np.eye(4)))


def test_coordinate_transformation():
    U1 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    U2 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    t1 = np.random.rand(3)*100
    t2 = np.random.rand(3)*100

    A = Frame.from_orign_orientation(origin=t1, orientation=U1)
    B = Frame.from_orign_orientation(origin=t2, orientation=U2)

    bTa = CoordinateTransformation.between_frames(A, B)

    coord_in_a = np.random.rand(3)
    coord_in_global = U1 @ coord_in_a + t1
    coord_in_b = U2.T @ (coord_in_global - t2)
    coord_in_b2 = bTa @ coord_in_a

    assert np.all(np.isclose(coord_in_b, coord_in_b2))


def test_coordinate_transformation():
    U1 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    U2 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    t1 = np.random.rand(3)*100
    t2 = np.random.rand(3)*100

    A = Frame.from_orign_orientation(origin=t1, orientation=U1)
    B = Frame.from_orign_orientation(origin=t2, orientation=U2)

    bTa = CoordinateTransformation.between_frames(A, B)

    A_to_global = mat4_from_translation_rotation(t1, U1)
    B_to_global = mat4_from_translation_rotation(t2, U2)

    coord_in_a = np.array([*np.random.rand(3), 1])
    coord_in_global = A_to_global @ coord_in_a
    coord_in_b = np.linalg.inv(B_to_global) @ coord_in_global
    coord_in_b2 = bTa @ coord_in_a

    assert np.all(np.isclose(coord_in_b, coord_in_b2))

def test_spatial_coordinates():
    U1 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    U2 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    t1 = np.random.rand(3)*100
    t2 = np.random.rand(3)*100

    A = Frame.from_orign_orientation(origin=t1, orientation=U1)
    B = Frame.from_orign_orientation(origin=t2, orientation=U2)
    bTa = CoordinateTransformation.between_frames(A, B)

    a_coords = np.random.rand(3)
    a_in_A = SpatialCoordinates(a_coords, A)
    a_in_B = a_in_A.express_in(B)
    assert np.all(np.isclose(a_in_B.coords, bTa @ a_coords))


def test_spatial_motion_coordinate_transform1():
    dx = np.array([1.0, 0.0, 0.0])
    ang = np.ones(3)
    lin = np.ones(3)
    m = SpatialMotion.from_angular_linear(ang, lin)
    m_prime = SpatialMotion.from_angular_linear(ang, lin + np.cross(ang, dx))

    A = Frame()
    B = Frame.orient_wrt(A, translation = dx)

    bTa = CoordinateTransformation.between_frames(A, B)
    m_prime_coords = bTa @ m

    assert np.all(np.isclose(m_prime.vec, m_prime_coords.vec))



def test_spatial_motion_coordinate_transforms():
    ang = np.ones(3)
    lin = np.ones(3)
    m = SpatialMotion.from_angular_linear(ang, lin)

    rotation = Rotation.from_euler('xyz', [23,45,-23], degrees=True).as_matrix()
    translation = np.random.rand(3) * 100

    G = Frame()
    A = Frame.orient_wrt(G, rotation, translation)

    # manually build up the transformation matrix
    transform = xlt(-translation) @ rot(rotation.T)
    m_prime = SpatialMotion(transform @ m.vec)
    m_coord_prime2 = SpatialCoordinates(m_prime, A)

    m_coord = SpatialCoordinates(m, G)
    m_coord_prime = m_coord.express_in(A)
    #assert m_coord_prime == m_coord_prime2




