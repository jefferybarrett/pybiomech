"""
This tests the spatial algebra part of the code
"""
import pytest
import numpy as np
from pybiomech.physics.spatial import *
from scipy.spatial.transform import Rotation


def test_spatial_inverse_inertia():
    c = 20 * (np.random.rand(3) - 0.5)
    mass = 12.34
    inertia = np.diag([1.0, 2.0, 3.0])

    spatial_inertia = SpatialInertia.from_mass_inertia_about_com(mass, inertia, c)
    inertia_inv = InverseInertia.from_mass_inertia_about_com(mass, inertia, c)

    assert np.all(np.isclose(inertia_inv.mat, spatial_inertia.inv().mat))
    assert inertia_inv.is_approximately(spatial_inertia.inv())


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
        m1 = SpatialMotion(v1.vec)
        f1 = SpatialForce(v2.vec)
        m2 = m1.copy()
        f2 = SpatialForce(v2.vec)
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
    I_inertia = SpatialMatrix(np.eye(6), domain = SpatialForce, range = SpatialForce)
    I_m = SpatialMatrix(np.eye(6), domain = SpatialMotion, range = SpatialMotion)
    I_f = SpatialMatrix(np.eye(6), domain = SpatialForce, range = SpatialForce)
    assert np.all(I @ v == v)
    assert np.all(I_m @ m == m)
    assert np.all(I_f @ f == f)
    assert np.all(I_inertia @ inertia == inertia)

def test_free_lin_decomposition():
    for _ in range(100):
        v = SpatialVector(100 * np.random.rand(6))
        free, line = v.free_lin_decompose()
        assert free.is_free()
        assert np.isclose(np.dot(line.angular, line.linear), 0.0)

def test_is_free():
    free_v = SpatialVector.from_angular_linear(np.zeros(3), np.ones(3))
    assert free_v.is_free()

    try:
        free, line = free_v.free_lin_decompose()
    except ZeroDivisionError as e:
        assert "is a Free Vector and cannot be decomposed in this way!" in str(e)
    else:
        assert False, "SpecificException was not raised!"

def test_decomposition():
    ang, lin = np.ones(3), np.ones(3)
    h = np.dot(ang, lin) / np.dot(ang, ang)
    ang_prime = ang
    lin_prime = lin - h * ang
    free_part = h * ang

    v = SpatialVector.from_angular_linear(ang, lin)
    free, line = v.free_lin_decompose()

    assert np.all(free.linear == free_part)
    assert np.all(line == SpatialVector.from_angular_linear(ang_prime, lin_prime))



def test_build():
    linear = 100 * np.random.rand(3)
    angular = 100 * np.random.rand(3)
    test = SpatialVector.from_angular_linear(angular, linear)
    assert np.all(linear == test.linear)
    assert np.all(angular == test.angular)
