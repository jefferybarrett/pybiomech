import numpy as np
from pybiomech.physics.spatial import *
from pybiomech.utils.math_utils import ZERO3, IDENTITY_3x3

class Frame:
    def __init__(self):
        self.mat4 = np.eye(4)
    
    @classmethod
    def from_orign_orientation(cls, origin = ZERO3, orientation = IDENTITY_3x3):
        mat4 = np.eye(4)
        mat4[:3, :3] = orientation
        mat4[:3, 3] = origin
        return cls(mat4)
    
    @classmethod
    def from_mat4(cls, mat4):
        assert np.shape(mat4) == (4,4)
        newframe = cls()
        newframe.mat4 = mat4
        return newframe


    @property
    def origin(self):
        """Returns the origin (translation vector) of the frame."""
        return self.mat4[:3, 3]


    @property
    def orientation(self):
        """Returns the orientation (direction cosine matrix) of the frame."""
        return self.mat4[:3, :3]

    
    def to_origin_orientation(self):
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

    