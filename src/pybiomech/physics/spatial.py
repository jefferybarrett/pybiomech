import numpy as np
from numbers import Number
from pybiomech.utils.math_utils import *

class SpatialVector:

    def __init__(self, vec=np.zeros(6)):
        assert len(vec) == 6, "A SpatialVector must have 6 components."
        self.vec = np.array(vec, dtype=float)
        self.domain = SpatialVector
    
    @classmethod
    def from_angular_linear(cls, angular, linear):
        """Create a SpatialVector from linear and angular components."""
        assert len(linear) == 3 and len(angular) == 3
        return cls(np.hstack((angular, linear)))
    
    @classmethod
    def ones(cls):
        return cls(np.array([1, 1, 1, 1, 1, 1]))

    @property
    def angular(self):
        return self.vec[0:3]
    
    @property
    def linear(self):
        return self.vec[3:]
    
    def to_angular_linear(self):
        return self.angular, self.linear
    
    def is_free(self):
        """In Featherstone's framework a free vector is one with no angular component.
        In other words it is a vector with just a linear component.
        """
        return np.all(self.angular == 0)
    
    def is_line_vector(self):
        """ In Featherstone's framework, a line vector is one whose linear and angular
        components are orthogonal to one another.
        """
        return (np.dot(self.linear, self.angular) == 0)
    
    def free_lin_decompose(self):
        if not (self.is_free()):
            h = np.dot(self.angular, self.linear) / np.dot(self.angular, self.angular)
            line_vector = type(self).from_angular_linear(self.angular, self.linear - h * self.angular)
            free_vector = type(self).from_angular_linear(ZERO3, h * self.angular)
            return  free_vector, line_vector
        raise ZeroDivisionError(f"{self.__repr__} is a Free Vector and cannot be decomposed in this way!")
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.vec + other.vec)
        elif isinstance(other, Number):
            return type(self)(self.vec + other)
        raise TypeError("Addition is only supported between two SpatialVectors.")

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.vec - other.vec)
        elif isinstance(other, Number):
            return type(self)(self.vec - other)
        raise TypeError("Addition is only supported between two SpatialVectors.")
    
    def __mul__(self, other):
        if isinstance(other, Number):
            return type(self)(self.vec * other)
        raise TypeError("Multiplication is only supported with a scalar.")
    
    def __abs__(self):
        return type(self)(np.abs(self.vec))
    
    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)

    def __repr__(self):
        return f"{type(self).__name__}({self.vec})"
    
    def __eq__(self, other):
        return (self.vec == other.vec) if isinstance(other, type(self)) else False
    
    def __ne__(self, other):
        return (self.vec != other.vec) if isinstance(other, type(self)) else True
    
    def copy(self):
        return type(self)(self.vec.copy())

    def is_approximately(self, other, rtol=1e-5, atol=1e-8):
        if not isinstance(other, type(self)):
            return False
        return np.allclose(self.vec, other.vec, rtol=rtol, atol=atol)

    def dot(self, other):
        if isinstance(other, SpatialVector):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product requires another SpatialVector.")

    def wedge(self):
        ang, lin = self.to_angular_linear()
        mat = np.block([[xprod_mat(ang), np.zeros((3,3))], [xprod_mat(lin), xprod_mat(ang)]])
        return SpatialMatrix(mat, domain = self.domain, range = type(self))

    def wedge_star(self):
        return -(self.wedge().T)

    def cross(self, other):
        return self.wedge() @ other

    def cross_star(self, other):
        return (self.wedge_star()) @ other

    def dot(self, other):
        if isinstance(other, self.domain):
            return np.dot(self.vec, other.vec)
        raise TypeError(f"Dot product for {type(self).__name__} must be with {self.domain.__name__}.")



class SpatialForce(SpatialVector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = SpatialMotion


class SpatialMotion(SpatialVector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = SpatialForce


class SpatialMatrix:
    def __init__(self, mat=np.eye(6), domain = SpatialVector, range = SpatialVector):
        assert mat.shape == (6, 6), "A SpatialMatrix must be a 6x6 matrix."
        self.mat = np.array(mat, dtype=float)
        self.domain = domain
        self.range = range

    def from_dyad(v1:SpatialVector, v2:SpatialVector):
        return SpatialMatrix(mat = v1.vec[:,np.newaxis] @ v2.vec[:,np.newaxis].T, domain = v2.domain, range = type(v1))
    
    @property
    def T(self):
        return SpatialMatrix(self.mat.T, domain = self.range, range = self.domain)

    def inv(self):
        matinv = np.linalg.inv(self.mat)
        return SpatialMatrix(matinv, domain = self.range, range = self.domain)

    def __add__(self, other):
        if isinstance(other, SpatialMatrix):
            return type(self)(self.mat + other.mat)
        raise TypeError(f"Addition not supported between {type(self).__name__} and {type(other).__name__}.")

    def __mul__(self, other):
        # note a SpatialMatrix will not change the type of vector it acts on
        # that is to say it sends M6 -> M6, F6 -> F6, and Inertia Tensors -> Inertia Tensors etc.
        if isinstance(other, (int, float)):
            return type(self)(self.mat * other)
    
    def __matmul__(self, other):
        if isinstance(other, self.domain):
            return self.range(self.mat @ other.vec)
        if isinstance(other, SpatialMatrix) and (other.range == self.domain):
            return SpatialMatrix(self.mat @ other.mat, range = self.range, domain = other.domain)
        raise TypeError(f"Multiplication not supported with type {type(other).__name__}.")
    
    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        
    def __neg__(self):
        return SpatialMatrix(-self.mat, domain = self.domain, range = self.range)

    def __repr__(self):
        return f"{type(self).__name__}(\n{self.mat}\n)"
    
    def __eq__(self, other):
        if issubclass(type(self), type(other)) and (other.domain == self.domain) and (other.range == self.range):
            return self.mat == other.mat
        else:
            return False
    
    def __ne__(self, other):
        return (self.mat != other.mat) if isinstance(other, type(self)) else True
    
    def is_approximately(self, other, rtol=1e-5, atol=1e-8):
        """Check if two SpatialMatrix instances are approximately equal."""
        if not isinstance(other, SpatialMatrix):
            return False
        return np.allclose(self.mat, other.mat, rtol=rtol, atol=atol) and (self.domain == other.domain) and (self.range == other.range)


class SpatialInertia(SpatialMatrix):

    def __init__(self, mat=np.eye(6)):
        super().__init__(mat, domain = SpatialMotion, range = SpatialForce)

    @classmethod
    def from_mass_inertia_about_com(cls, mass, inertia_tensor, c = ZERO3):
        """ This will construct a SpatialInertia where mass is its mass and inertia tensor
        is its inertia tensor evalauted about its centre of mass.
        """
        cx = xprod_mat(c)
        Ic = inertia_tensor + mass * (cx @ cx.T)
        diag_portion = mass * cx
        mat = np.block([[Ic, diag_portion], [-diag_portion, mass * np.eye(3)]])
        return cls(mat)

class InverseInertia(SpatialMatrix):

    def __init__(self, mat=np.eye(6)):
        super().__init__(mat, domain = SpatialForce, range = SpatialMotion)
    
    @classmethod
    def from_mass_inertia_about_com(cls, mass, inertia_tensor, c = ZERO3):
        cx = xprod_mat(c)
        Ic_inv = np.linalg.inv(inertia_tensor)
        mat = np.block([[Ic_inv, Ic_inv @ cx.T], [cx @ Ic_inv, (1/mass) * np.eye(3) + cx @ Ic_inv @ cx.T]])
        return cls(mat)





