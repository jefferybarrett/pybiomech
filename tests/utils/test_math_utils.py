import pytest
import numpy as np
from pybiomech.utils.math_utils import *
from scipy.spatial.transform import Rotation

def test_xprod_mat():
    for _ in range(100):
        u = (100 * (np.random.rand(3) - 0.5)).astype(int)
        v = 100 * (np.random.rand(3) - 0.5).astype(int)

        assert np.all(np.isclose(np.cross(u,v), xprod_mat(u) @ v))


def test_rot():
    E = Rotation.from_euler('xyz', np.random.rand(3)*np.pi*2).as_matrix()
    cero = np.zeros_like(E)
    assert np.all(rot(E) == np.block([[E, cero], [cero, E]]))
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    V = np.hstack([v1, v2])
    assert np.all(np.isclose(rot(E) @ V, np.hstack([E @ v1, E @ v2])))


def test_xlt():
    velocity = 100 * np.random.rand(3) - 50.0
    omega =  100 * (np.random.rand(3) - 0.5)
    r = 40 * (np.random.rand(3) - 0.5)
    
    # if we are considering the velocity of a point displaced from the origin then
    # this will be
    # v' = v_O + omega x r
    # here v_O is the velocity of the origin (that's velocity)
    # omega is the angular velocity of the body (also omega above)
    # r is the vector from the origin to the point
    # equivalently this is:
    # [omega', v'] = [omega, v_O - rx omega] [[1, 0], [-rx, 1]] [omega, v_O]
    # which gives us a nice independent way of testing this expression
    omega_prime = omega
    v_prime = velocity + np.cross(omega, r)
    
    res1 = np.hstack([omega_prime, v_prime])
    res2 = xlt(r) @ np.hstack([omega, velocity])
    assert np.all(np.isclose(res1, res2))



