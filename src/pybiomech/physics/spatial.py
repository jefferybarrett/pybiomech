import numpy as np
from numbers import Number
from pybiomech.utils.math_utils import *

# TODO: implement the cross product methods for SpatialForces and SpatialMotion

# maybe something like this is warranted? Not sure!
class SpatialVector:
    def __init__(self, vec=None):
        vec = np.zeros(6) if vec is None else vec
        assert len(vec) == 6, "A SpatialVector must have 6 components."
        self.vec = np.array(vec, dtype=float)

    @classmethod
    def from_angular_linear(cls, angular, linear):
        """Create a SpatialVector from linear and angular components."""
        assert len(linear) == 3 and len(angular) == 3
        return cls(np.hstack((angular, linear)))
    
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
        raise TypeError("Addition is only supported between two SpatialVectors.")

    def __mul__(self, other):
        if isinstance(other, Number):
            return type(self)(self.vec * other)
        raise TypeError("Multiplication is only supported with a scalar.")
    
    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        raise TypeError("Multiplication is only supported with a scalar.")

    def __repr__(self):
        return f"{type(self).__name__}({self.vec})"
    
    def __eq__(self, other):
        return (self.vec == other.vec) if isinstance(other, type(self)) else False
    
    def __ne__(self, other):
        return (self.vec != other.vec) if isinstance(other, type(self)) else True
    
    def copy(self):
        return type(self)(self.vec.copy())

    def is_approximately(self, other, rtol=1e-5, atol=1e-8):
        """Check if two SpatialMatrix instances are approximately equal."""
        if not isinstance(other, type(self)):
            return False
        return np.allclose(self.vec, other.vec, rtol=rtol, atol=atol)

    def dot(self, other):
        if isinstance(other, SpatialVector):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product requires another SpatialVector.")

    def wedge(self):
        ang, lin = self.to_angular_linear()
        return SpatialMatrix(np.block([[xprod_mat(ang), np.zeros((3,3))], [xprod_mat(lin), xprod_mat(ang)]]))

    def wedge_star(self):
        return -self.wedge().T

    def cross(self, other):
        if isinstance(other, SpatialVector):
            return self.wedge() @ other

    def cross_star(self, other):
        if isinstance(other, SpatialVector):
            return self.wedge_star() @ other

    def norm(self):
        return np.linalg.norm(self.vec)



class SpatialForce(SpatialVector):

    @classmethod
    def from_spatial(cls, vec:SpatialVector):
        return SpatialForce(vec.vec)

    def dot(self, other):
        if isinstance(other, SpatialMotion):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product only makes sense with a MotionVector.")


class SpatialMotion(SpatialVector):

    @classmethod
    def from_spatial(cls, sv:SpatialVector):
        return SpatialMotion(sv.vec)

    def dot(self, other):
        if isinstance(other, SpatialForce):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product donly makes sense with a WrenchVector.")



class SpatialMatrix:
    def __init__(self, mat=np.eye(6)):
        assert mat.shape == (6, 6), "A SpatialMatrix must be a 6x6 matrix."
        self.mat = np.array(mat, dtype=float)

    @property
    def T(self):
        return SpatialMatrix(self.mat.T)

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
        if isinstance(other, SpatialVector):
            return type(other)(self.mat @ other.vec)
        if isinstance(other, SpatialMatrix):
            return type(other)(self.mat @ other.mat)
        raise TypeError(f"Multiplication not supported with type {type(other).__name__}.")
    
    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        
    def __neg__(self):
        return SpatialMatrix(-self.mat)

    def __repr__(self):
        return f"{type(self).__name__}(\n{self.mat}\n)"
    
    def __eq__(self, other):
        return (self.mat == other.mat) if isinstance(other, type(self)) else False
    
    def __ne__(self, other):
        return (self.mat != other.mat) if isinstance(other, type(self)) else True
    
    def is_approximately(self, other, rtol=1e-5, atol=1e-8):
        """Check if two SpatialMatrix instances are approximately equal."""
        if not isinstance(other, SpatialMatrix):
            return False
        return np.allclose(self.data, other.data, rtol=rtol, atol=atol)


class SpatialInertia(SpatialMatrix):

    @classmethod
    def from_mass_inertia(cls, mass, inertia_tensor):
        """ This will construct a SpatialInertia where mass is its mass and inertia tensor
        is its inertia tensor evalauted about its centre of mass.
        """
        mat = np.block([[inertia_tensor, np.zeros((3,3))], [np.zeros((3,3)), mass * np.eye(3)]])
        return SpatialInertia(mat)
    
    def __matmul__(self, other):
        if isinstance(other, SpatialMotion):
            return SpatialForce(self.mat @ other.vec)
        if isinstance(other, SpatialForce):
            raise TypeError("SpatialInertia must act on a SpatialMotion type object")






