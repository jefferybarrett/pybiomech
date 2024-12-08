import numpy as np
from pybiomech.physics.spatial import *
from pybiomech.utils.math_utils import ZERO3, IDENTITY_3x3

# perhaps this is the way to go.
import numpy as np

class Frame:
    def __init__(self, mat4=None, velocity:SpatialMotion = SpatialMotion()):
        """
        Creates a frame using a 4x4 transformation matrix.
        If no matrix is provided, defaults to an identity matrix.
        """
        self.mat4 = mat4 if mat4 is not None else np.eye(4)
        self.v0 = velocity

    @property
    def origin(self):
        """Returns the origin (translation vector) of the frame."""
        return self.mat4[:3, 3]

    @property
    def orientation(self):
        """Returns the orientation (direction cosine matrix) of the frame."""
        return self.mat4[:3, :3]

    @classmethod
    def from_components(cls, origin = ZERO3, orientation = IDENTITY_3x3):
        """
        Creates a frame from an origin and an orientation matrix.
        :param origin: A 3-element array-like representing the translation vector.
        :param orientation: A 3x3 matrix representing the direction cosine matrix.
        """
        mat4 = np.eye(4)
        mat4[:3, :3] = orientation
        mat4[:3, 3] = origin
        return cls(mat4)

    def to_components(self):
        """Returns the origin and orientation as separate components."""
        return self.origin, self.orientation

    def inv(self):
        """Inverts the frame (computes the inverse transformation)."""
        inv_mat4 = np.eye(4)
        inv_mat4[:3, :3] = self.orientation.T  # Transpose of the rotation matrix
        inv_mat4[:3, 3] = -self.orientation.T @ self.origin  # Inverted translation
        return Frame(inv_mat4)

    def __matmul__(self, other):
        """
        Multiplies (composes) this frame with another.
        This corresponds to applying the transformation of this frame to another.
        """
        if isinstance(other, Frame):
            return Frame(self.mat4 @ other.mat4)
        raise ValueError("Can only multiply Frame with another Frame.")


class CoordinateTransformation(SpatialMatrix):
    
    def __init__(self, rotation = np.eye(3), translation = np.zeros(3)):
        self.rotation = rotation
        self.r = translation
        self.mat = mat4_from_components(rotation, translation)
    
    @classmethod
    def between_frames(cls, _from: Frame, _to:Frame):
        dx, E = (_to.inv() @ _from).to_components()
        return cls(rotation=E, translation=dx)
    
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
    
    def inv(self):
        return CoordinateTransformation(self.rotation.T, translation= -self.rotation.T @ self.r)
    
    def __matmul__(self, other):
        if isinstance(other, SpatialMotion):
            return SpatialMotion(self.motionTransform @ other.vec)
        elif isinstance(other, SpatialForce):
            return SpatialForce(self.forceTransform @ other.vec)
        elif isinstance(other, SpatialInertia):
            return SpatialInertia(self.forceTransform @ other.mat @ self.motionInverse)
        elif isinstance(other, np.ndarray) and np.shape(other)[0] == 4:
            return self.mat @ other
        elif isinstance(other, np.ndarray) and np.shape(other)[0] == 3:
            return self.rotation @ other + self.r
        elif isinstance(other, CoordinateTransformation): # composition
            return CoordinateTransformation(*mat4_to_components(self.mat @ other.mat))
        raise TypeError(f"Coordinate Transforms must act on SpatialForces or SpatialMotions and not {type(other).__name__}.")



if __name__ == "__main__":
    print("testing this module!")
    from scipy.spatial.transform import Rotation
    E = Rotation.from_euler('x', 45, degrees = True).as_matrix()
    
    G = Frame() # ground frame
    A = Frame.from_components(orientation = E)
    transform = CoordinateTransformation.between_frames(_from=G, _to=A)
    print(transform)
    