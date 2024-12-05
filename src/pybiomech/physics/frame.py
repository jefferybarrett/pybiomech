import numpy as np


# perhaps this is the way to go.
class Frame:
    def __init__(self, origin:np.array = np.zeros(3), orientation:np.array = np.eye(3)):
        """ Creates a frame with a given orientation (direction cosine matrix)
        and position with respect to a background (lab based) frame.
        """
        self.origin = origin
        self.orientaiton = orientation

    def local2global(self, point:np.array) -> np.array:
        """ Get the global position of a local point.

        """
        return self.orientaiton @ point + self.origin
    
    def global2local(self, point):
        """ Get the local representation of a global point.
        """
        return self.orientaiton.T @ (point - self.origin)
    
    def Mat4(self):
        raise NotImplementedError("Potentially could wait until pyglet integration")


if __name__ == "__main__":
    print("testing this module!")
    from scipy.spatial.transform import Rotation

    G = Frame() # ground frame
    A = Frame(
        orientation = Rotation.from_euler('x', 24, degrees = True)
    )
    