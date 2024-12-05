import numpy as np
from numbers import Number
from pybiomech.physics.frame import Frame


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
        return f"SpatialVector({self.vec})"
    
    def dot(self, other):
        if isinstance(other, SpatialVector):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product requires another SpatialVector.")

    def cross(self, other):
        raise NotImplementedError("Cross product is only defined with SpatialMotion and SpatialForce vectors")

    def wedge(self):
        raise NotImplementedError

    def norm(self):
        return np.linalg.norm(self.vec)


    
class SpatialForce(SpatialVector):
    def dot(self, other):
        if isinstance(other, SpatialMotion):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product only makes sense with a MotionVector.")

    def wedge(self):
        raise NotImplementedError("Working on this!")

    def cross(self, other):
        if isinstance(other, SpatialForce):
            return self.wedge @ other
        raise TypeError("Spatial cross product only makes sense with a SpatialForce")


class SpatialMotion(SpatialVector):
    def dot(self, other):
        if isinstance(other, SpatialForce):
            return np.dot(self.vec, other.vec)
        raise TypeError("Dot product donly makes sense with a WrenchVector.")




class SpatialMatrix:
    def __init__(self, mat=np.eye(6)):
        assert mat.shape == (6, 6), "A SpatialMatrix must be a 6x6 matrix."
        self.mat = np.array(mat, dtype=float)

    def __add__(self, other):
        if isinstance(other, SpatialMatrix):
            return type(self)(self.mat + other.mat)
        raise TypeError(f"Addition not supported between {type(self).__name__} and {type(other).__name__}.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(self.mat * other)
        if isinstance(other, SpatialVector):
            return SpatialVector(self.mat @ other.vec) # we will note that this will change the outcome for an inertia matrix
        raise TypeError(f"Multiplication not supported with type {type(other).__name__}.")
    
    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)

    def __repr__(self):
        return f"{type(self).__name__}(\n{self.mat}\n)"


class CoordinateTransformation:
    
    def __init__(self, rotation = np.eye(3), translation = np.zeros(3)):
        self.rotation = rotation
        self.r = translation

    @property
    def motionTransform(self):
        zeromat = np.zeros((3,3))
        rx, ry, rz = self.r
        xprod_mat = np.array([
            [0, -rz, ry],   # First row
            [rz, 0, -rx],   # Second row
            [-ry, rx, 0]    # Third row
        ])
        rot6x6 = np.vstack([np.hstack([self.rotation,zeromat]), np.hstack([zeromat, self.rotation])])
        xprod_part = np.vstack([np.hstack([np.eye(3), zeromat]), np.hstack([-xprod_mat, np.eye(3)])])
        return rot6x6 @ xprod_part
    
    @property
    def forceTransform(self):
        zeromat = np.zeros((3,3))
        rx, ry, rz = self.r
        xprod_mat = np.array([
            [0, -rz, ry],
            [rz, 0, -rx],
            [-ry, rx, 0]  
        ])
        rot6x6 = np.vstack([np.hstack([self.rotation,zeromat]), np.hstack([zeromat, self.rotation])])
        xprod_part = np.vstack([np.hstack([np.eye(3), -xprod_mat]), np.hstack([zeromat, np.eye(3)])])
        return rot6x6 @ xprod_part
    
    def __matmul__(self, other):
        if isinstance(other, SpatialMotion):
            return SpatialMotion(self.motionTransform @ other.vec)
        elif isinstance(other, SpatialForce):
            return SpatialForce(self.forceTransform @ other.vec)
        raise TypeError(f"Coordinate Transforms must act on SpatialForces or SpatialMotions and not {type(other).__name__}.")
            



if __name__ == "__main__":
    print("Testing this module!")
    force = SpatialForce.from_linear_angular([0, 0, 10], [0, 0, 1])
    motion = SpatialMotion.from_linear_angular([0, 0, 10], [0, 0, 1])
    X = CoordinateTransformation(translation = np.array([0, 1, 1]))
    
    print(X @ motion)



