import numpy as np
from pybiomech.physics.spatial import *
from pybiomech.physics.frame import Frame

class SpatialCoordinates:

    def __init__(self, spatial:SpatialVector|SpatialMatrix, frame:Frame):
        self.coords = spatial
        self.frame = frame

    def express_in(self, target_frame:Frame):
        transform = CoordinateTransformation.between_frames(self.frame, target_frame)
        return SpatialCoordinates(transform @ self.coords, frame = target_frame)

    def __repr__(self):
        return f"{self.coords.__repr__()} expressed in {self.frame.__repr__()}"
    
    def __eq__(self, other):
        if isinstance(other, SpatialCoordinates) and (self.frame == other.frame):
            return np.all(self.coords == other.coords)
        elif isinstance(other, SpatialCoordinates) and (self.frame != other.frame):
            return self == other.express_in(self.frame)
        else:
            return False



class CoordinateTransformation:
    
    def __init__(self, translation = np.zeros(3), rotation = np.eye(3)):
        self.rotation = rotation
        self.translation = translation
        self.mat = mat4_from_translation_rotation(translation, rotation)
    
    @classmethod
    def between_frames(cls, from_frame: Frame, to_frame:Frame):
        dx, E = (to_frame.inv() @ from_frame).as_origin_orientation()
        return cls(rotation=E, translation=dx)
    
    def transform_vector(self, vector:SpatialVector):
        if (isinstance(vector, SpatialForce)):
            return self.forceTransform @ vector
        else:
            return self.motionTransform @ vector

    def transform_matrix(self, matrix:SpatialMatrix):
        left_side = self.forceTransform if matrix.range == SpatialForce else self.motionTransform
        right_side = self.forceInverse if matrix.domain == SpatialForce else self.motionInverse
        return (left_side @ matrix) @ right_side

    @property
    def motionTransform(self):
        return SpatialMatrix(rot(self.rotation) @ xlt(self.translation), domain = SpatialMotion, range = SpatialMotion)
    
    @property
    def motionInverse(self):
        return SpatialMatrix(xlt(-self.translation) @ rot(self.rotation.T), domain = SpatialMotion, range = SpatialMotion)

    @property
    def forceTransform(self):
        return SpatialMatrix(rot(self.rotation) @ xlt(-self.translation).T, domain = SpatialForce, range = SpatialForce)
    
    @property
    def forceInverse(self):
        return SpatialMatrix(xlt(self.translation).T @ rot(self.rotation.T), domain = SpatialForce, range = SpatialForce)
    
    def inv(self):
        return CoordinateTransformation(-self.rotation.T @ self.translation, self.rotation.T)
    
    def __matmul__(self, other):
        if isinstance(other, SpatialVector):
            return self.transform_vector(other)
        elif isinstance(other, SpatialMatrix):
            return self.transform_matrix(other)
        elif isinstance(other, np.ndarray) and np.shape(other)[0] == 4:
            return self.mat @ other
        elif isinstance(other, np.ndarray) and np.shape(other)[0] == 3:
            return self.rotation @ other + self.translation
        elif isinstance(other, CoordinateTransformation): # composition
            return CoordinateTransformation(*mat4_to_translation_rotation(self.mat @ other.mat))
        raise TypeError(f"Coordinate Transforms must act on SpatialForces or SpatialMotions and not {type(other).__name__}.")
    
    def __repr__(self):
        return f"Coordinate Transformation with {self.mat.__repr__}"
