"""
This tests the spatial algebra part of the code
"""
import pytest
import numpy as np
from pybiomech.physics.spatial import *
from scipy.spatial.transform import Rotation


def test_matrix_negate():
    for _ in range(100):
        A = 100 * (np.random.rand(6,6) - 0.5)
        assert np.all(-SpatialMatrix(A) == SpatialMatrix(-A))

def test_matrix_transpose():
    mat = 100 * (np.random.rand(6, 6) - 0.5)
    assert np.all(SpatialMatrix(mat).T == SpatialMatrix(mat.T))

def test_matrix_equality():
    for _ in range(100):
        A = 100 * (np.random.rand(6,6) - 0.5)
        assert np.all(SpatialMatrix(A) != A)
        assert np.all((SpatialMatrix(A) != SpatialMatrix(A + 0.1)))
        assert np.all((SpatialMatrix(A) != SpatialInertia(A)))

def test_spatial_arithmetic():
    for _ in range(100):
        xlin = (100 * np.random.rand(3) - 50).astype(int)
        xrot = (100 * np.random.rand(3) - 50).astype(int)
        ylin = (300 * np.random.rand(3)).astype(int)
        yrot = (300 * np.random.rand(3)).astype(int)

        # constants for multiplication
        alpha, beta = np.random.random(2)

        # addition should be linear
        v1 = SpatialVector.from_angular_linear(xrot, xlin)
        v2 = SpatialVector.from_angular_linear(yrot, ylin)
        lin_combo = alpha * v1 + beta * v2
        other_lin_combo = SpatialVector.from_angular_linear(alpha * xrot + beta * yrot, alpha * xlin + beta * ylin)
        assert np.all(lin_combo == other_lin_combo)

        # spatial motions and spatial forces are distinct from spatial vectors
        m1 = SpatialMotion.from_spatial(v1)
        f1 = SpatialForce.from_spatial(v2)
        m2 = m1.copy()
        f2 = SpatialForce.from_spatial(v2)
        assert np.all(m1 != v1)
        assert np.all(f1 != v2)
        assert np.all(m1 == m2)
        assert np.all(f2 != m2)

        # now for some known cases
        assert m1.dot(f2) == xlin.dot(ylin) + xrot.dot(yrot)
        assert m1.dot(f2) == f2.dot(m1)



def test_spatial_matrix_types():
    """
    A SpatialMatrix maps M6 -> M6, F6 -> F6 and doesn't change type of a SpatialMatrix
    """
    v = SpatialVector(100 * np.random.rand(6))
    f = SpatialForce(100 * np.random.rand(6))
    m = SpatialMotion(100 * np.random.rand(6))
    inertia = SpatialInertia(np.random.rand(6, 6))
    
    I = SpatialMatrix(np.eye(6))
    assert np.all(I @ v == v)
    assert np.all(I @ m == m)
    assert np.all(I @ f == f)
    assert np.all(I @ inertia == inertia)

def test_free_lin_decomposition():
    for _ in range(100):
        v = SpatialVector(100 * np.random.rand(6))
        free, line = v.free_lin_decompose()
        assert free.is_free()
        assert np.isclose(np.dot(line.angular, line.linear), 0.0)
        

def test_build():
    linear = 100 * np.random.rand(3)
    angular = 100 * np.random.rand(3)
    test = SpatialVector.from_angular_linear(angular, linear)
    assert np.all(linear == test.linear)
    assert np.all(angular == test.angular)