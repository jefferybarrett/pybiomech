import numpy
from pybiomech.physics.frame import *
from pybiomech.physics.spatial import *
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