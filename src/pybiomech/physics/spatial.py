import numpy as np
from numbers import Number
from pybiomech.physics.frame import Frame
from pybiomech.utils.math_utils import xprod_mat, xlt, rot

# maybe something like this is warranted? Not sure!
class SpatialVector:
    def __init__(self, vec=None, frame = Frame()):
        vec = np.zeros(6) if vec is None else vec
        assert len(vec) == 6, "A SpatialVector must have 6 components."
        self.vec = np.array(vec, dtype=float)
        self.frame = frame

    @classmethod
    def from_linear_angular(cls, linear, angular):
        """Create a SpatialVector from linear and angular components."""
        assert len(linear) == 3 and len(angular) == 3
        return cls(np.hstack((angular, linear)))
    
    def to_linear_angular(self):
        angular, linear = self.vec[0:3], self.vec[3:]
        return linear, angular
    
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
        return isinstance(other, type(self)) and np.all(self.vec == other.vec)
    
    def copy(self):
        return type(self)(self.vec.copy())

    def is_approximately(self, other, rtol=1e-5, atol=1e-8):
        """Check if two SpatialMatrix instances are approximately equal."""
        if not isinstance(other, SpatialMatrix):
            return False
        return np.allclose(self.data, other.data, rtol=rtol, atol=atol)

    def dot(self, other):
        if isinstance(other, SpatialVector):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product requires another SpatialVector.")

    def wedge(self):
        lin, ang = self.to_linear_angular()
        return SpatialMatrix(np.block([[xprod_mat(ang), np.zeros((3,3))], [xprod_mat(lin), xprod_mat(ang)]]))

    def wedge_star(self):
        return -self.wedge().T

    def cross(self, other):
        raise NotImplementedError("Cross product is only defined with SpatialMotion and SpatialForce vectors")

    def cross_star(self, other):
        raise NotImplementedError("Still working on this!")

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
        if isinstance(other, (int, float)):
            return type(self)(self.mat * other)
        if isinstance(other, SpatialVector):
            return SpatialVector(self.mat @ other.vec) # we will note that this will change the outcome for an inertia matrix
        if isinstance(other, SpatialMatrix):
            return SpatialMatrix(self.mat @ other.mat)
        raise TypeError(f"Multiplication not supported with type {type(other).__name__}.")
    
    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        
    def __neg__(self):
        return SpatialMatrix(-self.mat)

    def __repr__(self):
        return f"{type(self).__name__}(\n{self.mat}\n)"
    
    def __eq__(self, other):
        return isinstance(other, type(self)) and np.all(self.mat == other.mat)
    
    def is_approximately(self, other, rtol=1e-5, atol=1e-8):
        """Check if two SpatialMatrix instances are approximately equal."""
        if not isinstance(other, SpatialMatrix):
            return False
        return np.allclose(self.data, other.data, rtol=rtol, atol=atol)


class SpatialInertia(SpatialMatrix):

    @classmethod
    def from_mass_inertia(cls, mass, inertia_tensor):
        mat = np.block([[inertia_tensor, np.zeros((3,3))], [np.zeros((3,3)), mass * np.eye(3)]])
        return SpatialInertia(mat)

class CoordinateTransformation(SpatialMatrix):
    
    def __init__(self, rotation = np.eye(3), translation = np.zeros(3)):
        self.rotation = rotation
        self.r = translation
        self.mat = self.motionTransform

    def get_transformation_from(A : Frame, to: Frame):
        rel_rotation = to.orientaiton.T @ A.orientaiton
        rel_displacement = to.origin - A.origin
        return CoordinateTransformation(rel_rotation, rel_displacement)

    @property
    def motionTransform(self):
        return rot(self.rotation) @ xlt(self.r)
    
    @property
    def motionInverse(self):
        return xlt(-self.r) @ rot(self.rotation.T)

    @property
    def forceTransform(self):
        return rot(self.rotation) @ xlt(-self.r).T
    
    @property
    def forceInverse(self):
        return xlt(self.r).T @ rot(self.rotation.T)
    
    def __matmul__(self, other):
        if isinstance(other, SpatialMotion):
            return SpatialMotion(self.motionTransform @ other.vec)
        elif isinstance(other, SpatialForce):
            return SpatialForce(self.forceTransform @ other.vec)
        elif isinstance(other, SpatialMatrix):
            raise NotImplementedError("There is probably a rule for transforming a spatial matrix, it is not yet implemented.")
        raise TypeError(f"Coordinate Transforms must act on SpatialForces or SpatialMotions and not {type(other).__name__}.")
    

if __name__ == "__main__":
    print("Testing this module!")
    from scipy.spatial.transform import Rotation

    force = SpatialForce.from_linear_angular(np.random.rand(3), np.random.rand(3))
    motion = SpatialMotion.from_linear_angular(np.random.rand(3), np.random.rand(3))



    print(force)
    print(force.wedge_star())



