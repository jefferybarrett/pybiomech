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
    T = CoordinateTransformation.between_frames(A, G)
    assert np.all(np.isclose(T.mat, mat4))

def test_frame_construction2():
    rotation = Rotation.from_euler('xyz', [25, 32, 35], degrees=True).as_matrix()
    translation = np.array([1.0, 2.0, -2.3])
    
    A = Frame()
    B = Frame.orient_wrt(A, rotation, translation)
    b2a = CoordinateTransformation.between_frames(from_frame = B, to_frame = A)
    assert np.all(b2a.translation == translation)
    assert np.all(b2a.rotation == rotation)



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

    A2B = CoordinateTransformation.between_frames(A, B)

    coord_in_a = 100*(np.random.rand(3) - 0.5)
    coord_in_global = U1 @ coord_in_a + t1
    coord_in_b = U2.T @ (coord_in_global - t2)
    coord_in_b2 = A2B @ coord_in_a

    assert np.all(np.isclose(coord_in_b, coord_in_b2))

def test_coordinate_transformation2():
    U1 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    U2 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    t1 = np.random.rand(3)*100
    t2 = np.random.rand(3)*100

    G = Frame() # global coordinate system
    A = Frame.from_orign_orientation(origin=t1, orientation=U1)
    B = Frame.from_orign_orientation(origin=t2, orientation=U2)
    

    coord_in_a = 100*(np.random.rand(3) - 0.5)
    vec_in_a = SpatialCoordinates(coord_in_a, A)
    coord_in_global = vec_in_a.express_in(G).coords
    coord_in_b = vec_in_a.express_in(B).coords
    
    coord_in_global2 = A.orientation @ coord_in_a + A.origin
    coord_in_b2 = B.orientation.T @ (coord_in_global2 - B.origin)

    assert np.all(np.isclose(coord_in_b, coord_in_b2))
    assert np.all(np.isclose(coord_in_global, coord_in_global2))



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

def test_coordinate_transformation2():
    U1 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    U2 = Rotation.from_euler('xyz', np.random.rand(3)*2*np.pi).as_matrix()
    t1 = np.random.rand(3)*100

    # these two frames have a common origin
    A = Frame.from_orign_orientation(origin=t1, orientation=U1)
    B = Frame.from_orign_orientation(origin=t1, orientation=U2)
    
    # the relative rotation between these two is:
    bEa = U2.T @ U1
    
    # this transforms from A to B coordinates
    bTa = CoordinateTransformation.between_frames(A, B)
    
    # double check that this transformation still works
    assert np.all(np.isclose(rot(bEa), bTa.motionTransform.mat))
    
    # now let m be a SpatialMotion vector, expressed in Frame A
    m = SpatialCoordinates(SpatialMotion(np.random.rand(6)), A)
    
    # now we will express them in Frame B
    m_in_b = m.express_in(B)
    
    # this subindexing is getting ridiculous lol (but hopefully with the abstraction layer this is not necessary)
    assert np.all(np.isclose((SpatialMatrix(rot(bEa)) @ m.coords).vec, m_in_b.coords.vec))



def test_spatial_motion_coordinate_transform1():
    dx = np.array([1.0, 3.0, -2.0])
    ang = np.ones(3)
    lin = np.ones(3)
    m = SpatialMotion.from_angular_linear(ang, lin)
    m_prime = SpatialMotion.from_angular_linear(ang, lin - np.cross(dx, ang))

    O = Frame.from_orign_orientation(origin = np.array([0.0, 0.0, 1.0]))
    P = Frame.orient_wrt(O, translation = dx)

    bTa = CoordinateTransformation.between_frames(O, P)
    m_prime_coords = bTa @ m

    assert np.all(np.isclose(m_prime.vec, m_prime_coords.vec))


def test_spatial_motion_coordinate_transforms():
    U = Rotation.from_euler('xyz', [23,45,-23], degrees=True).as_matrix()
    rotation = Rotation.from_euler('xyz', [23,45,-23], degrees=True).as_matrix()
    translation = np.random.rand(3) * 100
    O =  50 * (np.random.rand(3) - 0.5)
    P = O + U @ translation

    A = Frame.from_orign_orientation(O, U)
    B = Frame.orient_wrt(A, rotation, translation)
    assert np.all(np.isclose(B.origin, P)), "Origin of frame B is not at point P"
    
    # is the translation from A to B in A's coordinates. That's really good to know.
    
    # create the coordinates of m in A
    ang = np.ones(3)
    lin = np.ones(3)
    m_in_A = SpatialMotion.from_angular_linear(ang, lin)

    # manually build up the transformation matrix
    # then use formula 2.24 in Featherstone to get the new coordinates
    # we need:
    # OP in A's coordinate system, this is the vector r:
    # the rotation matrix that converts from A to B, that's B.orientation.T @ A.orientation
    r = U.T @ (P - O)                       # is equal to translation
    E = B.orientation.T @ A.orientation     # is equalt to rotation.T
    transform = rot(E) @ xlt(r)
    m_in_B = SpatialMotion(transform @ m_in_A.vec)
    m_expressed_B = SpatialCoordinates(m_in_B, B)

    m_expressed_A = SpatialCoordinates(m_in_A, A)
    m_reexpressed_B = m_expressed_A.express_in(B)
    assert np.all(np.isclose(m_reexpressed_B.coords.vec, m_expressed_B.coords.vec))


def test_spatial_motion_coordinate_transforms2():
    U = Rotation.from_euler('xyz', [23,45,-23], degrees=True).as_matrix()
    rotation = Rotation.from_euler('xyz', [23,45,-23], degrees=True).as_matrix()
    translation = np.random.rand(3) * 100
    O =  50 * (np.random.rand(3) - 0.5)
    P = O + U @ translation

    A = Frame.from_orign_orientation(O, U)
    B = Frame.orient_wrt(A, rotation, translation)
    assert np.all(np.isclose(B.origin, P)), "Origin of frame B is not at point P"
    
    # the conversion between these two frames
    A2B = B.orientation.T @ A.orientation
    r_in_B = A2B @ translation
    
    # create the coordinates of m in A
    ang_A = np.ones(3)
    lin_A = np.ones(3)
    m_in_A = SpatialMotion.from_angular_linear(ang_A, lin_A)
    
    # now let's manually change between frames using normal 3D vectors
    ang_B = A2B @ ang_A
    lin_B = A2B @ lin_A + np.cross(ang_B, r_in_B)
    m_in_B = SpatialMotion.from_angular_linear(ang_B, lin_B)
    
    # create the coordinate vector
    motion_vec = SpatialCoordinates(m_in_A, A)
    m_in_B2 = motion_vec.express_in(B).coords
    assert np.all(np.isclose(m_in_B.vec, m_in_B2.vec))



def test_spatial_force_coordinate_transforms():
    U = Rotation.from_euler('xyz', [23,45,-23], degrees=True).as_matrix()
    rotation = Rotation.from_euler('xyz', [23,45,-23], degrees=True).as_matrix()
    translation = np.random.rand(3) * 100
    O =  50 * (np.random.rand(3) - 0.5)
    P = O + U @ translation

    A = Frame.from_orign_orientation(O, U)
    B = Frame.orient_wrt(A, rotation, translation)
    assert np.all(np.isclose(B.origin, P)), "Origin of frame B is not at point P"
    
    # is the translation from A to B in A's coordinates. That's really good to know.
    A2B = B.orientation.T @ A.orientation
    r_in_B = A2B @ translation
    
    # create the coordinates of m in A
    torque_A = np.ones(3)
    force_A = np.ones(3)
    f_in_A = SpatialForce.from_angular_linear(torque_A, force_A)

    # let's very manually change the coordinate system here
    force_B = A2B @ force_A
    torque_B = A2B @ torque_A - np.cross(r_in_B, force_B)
    f_in_B = SpatialForce.from_angular_linear(torque_B, force_B)
    
    # now let's build the spatial coordinate and test the coords
    f_coords = SpatialCoordinates(f_in_A, A)
    f_in_B2 = f_coords.express_in(B).coords 
    assert np.all(np.isclose(f_in_B.vec, f_in_B2.vec))
    
def test_inertia_transform():
    dx = np.array([-1.0, 0.0, 0.0])
    A = Frame()
    B = Frame.orient_wrt(A, translation = dx)
    
    # assume we just have a sphere
    mass = 10.23
    radius = np.pi/8 # m
    inertia = (2.0/3.0) * mass * radius**2 * np.eye(3)
    I_in_A = SpatialInertia.from_mass_inertia_about_com(mass, inertia)
    
    inertia_prime = inertia + mass * (xprod_mat(-dx) @ xprod_mat(-dx).T)
    off_diagonal = mass * xprod_mat(-dx)
    coords_in_B = np.block([[inertia_prime, off_diagonal], [-off_diagonal, mass * IDENTITY_3x3]])
    assert inertia_prime[1,1] == inertia[1,1] + mass * (dx[0]**2 + dx[2]**2)
    I_in_B = SpatialInertia(coords_in_B) #.from_mass_inertia_about_com(mass_prime, inertia_prime)
    
    # this should obey the parallel axis theorem
    I_vec = SpatialCoordinates(I_in_A, A)
    I_in_B2 = I_vec.express_in(B).coords

    assert np.all(np.isclose(I_in_B2.mat, coords_in_B))


if __name__ == "__main__":
    test_inertia_transform()
