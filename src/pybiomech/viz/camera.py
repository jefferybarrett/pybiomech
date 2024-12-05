import numpy as np
from .constants import *
import pyglet
from pyglet.math import Mat4

class Camera(object):
    
    def __init__(
        self,
        pos = np.array([0.0, 0.0, 1.0]),
        focal_point = ORIGIN,
        azimuth = 0.0,
        elevation = 0.0,
        fov = 60.0,
        aspect_ratio = 16/9,
        near_clip = 0.1,
        far_clip = 100.0,
    ):
        self.pos = pos
        self.focal_point = focal_point
        self.azimuth = np.radians(azimuth)
        self.elevation = np.radians(elevation)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near_clip = near_clip
        self.far_clip = far_clip
        

    def get_view_matrix(self):
        """
        Computes the view matrix for the camera, 
        considering azimuth, elevation, and focal point.
        """
        # Compute spherical coordinates
        radius = np.linalg.norm(self.pos - self.focal_point)
        x = radius * np.cos(self.elevation) * np.cos(self.azimuth)
        y = radius * np.cos(self.elevation) * np.sin(self.azimuth)
        z = radius * np.sin(self.elevation)
        camera_position = self.focal_point + np.array([x, y, z])

        # Compute forward, right, and up vectors
        forward = (self.focal_point - camera_position)
        forward /= np.linalg.norm(forward)

        up = np.array([0.0, 0.0, 1.0])  # Assuming Z is up
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        # Construct the view matrix
        view_matrix = Mat4.look_at(
            eye=(camera_position[0], camera_position[1], camera_position[2]),
            target=(self.focal_point[0], self.focal_point[1], self.focal_point[2]),
            up=(up[0], up[1], up[2])
        )
        return view_matrix
