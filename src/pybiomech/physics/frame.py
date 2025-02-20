import numpy as np
from pybiomech.physics.spatial import *
from pybiomech.utils.math_utils import ZERO3, IDENTITY_3x3

class Frame:
    def __init__(self):
        self._mat4 = np.eye(4)
    
    @classmethod
    def orient_wrt(cls, reference:'Frame', rotation = IDENTITY_3x3, translation = ZERO3):
        """Orient a new Frame relative to an existing frame.
        rotation and translation specify the transformation that goes from the new frame to 
        the reference frame. Specifcally we will recover this matrix if we use
        
        Parameters:
        ----------
            reference: Frame
                A reference Frame which to base the new frame off of
            rotation: 3x3 matrix (default is the identity)
                The relative rotation from the new frame to the existing one
            translation: 3x1 vector (default is zero)
                The relative translation from the new frame to the existing one
        
        Example:
            A = Frame()
            B = Frame.orient_wrt(A, rotation, translation)
            b2a = CoordinateTransformation.between_frames(from_frame = B, to_frame = A)
            # b2a.translation == translation
            # b2a.rotation == rotation
        """
        new_origin = reference.origin + reference.orientation @ translation
        new_orientation = reference.orientation @ rotation
        return cls.from_orign_orientation(new_origin, new_orientation)

    @classmethod
    def from_orign_orientation(cls, origin = ZERO3, orientation = IDENTITY_3x3):
        mat4 = np.eye(4)
        mat4[:3, :3] = orientation
        mat4[:3, 3] = origin
        return cls.from_mat4(mat4)
    
    @classmethod
    def from_mat4(cls, mat4):
        assert np.shape(mat4) == (4,4)
        newframe = cls()
        newframe._mat4 = mat4
        return newframe

    @property
    def origin(self):
        """Returns the origin (translation vector) of the frame."""
        return self._mat4[:3, 3]

    @property
    def orientation(self):
        """Returns the orientation (direction cosine matrix) of the frame."""
        return self._mat4[:3, :3]
    
    
    def as_origin_orientation(self):
        return self.origin, self.orientation

    def as_mat4(self):
        return self._mat4

    def inv(self):
        """Inverts the frame (computes the inverse transformation)."""
        inv_mat4 = np.eye(4)
        inv_mat4[:3, :3] = self.orientation.T  # Transpose of the rotation matrix
        inv_mat4[:3, 3] = -self.orientation.T @ self.origin  # Inverted translation
        return Frame.from_mat4(inv_mat4)

    def __matmul__(self, other):
        """
        Multiplies (composes) this frame with another.
        This corresponds to applying the transformation of this frame to another.
        """
        if isinstance(other, Frame):
            return Frame.from_mat4(self._mat4 @ other._mat4)
        if isinstance(other, np.ndarray) and np.shape(other)[0] == 4:
            return self._mat4 @ other
        raise ValueError("Can only multiply Frame with another Frame.")
    
    def __eq__(self, other):
        if isinstance(other, Frame):
            return np.all(self.as_mat4() == other.as_mat4())
        return False
    
    def is_approx(self, other:'Frame'):
        return np.all(np.isclose(self.as_mat4(), other.as_mat4()))


    