import numpy
from pybiomech.physics.frame import *
from pybiomech.physics.spatial import *
from scipy.spatial.transform import Rotation



def test_frame_construct():
    R = Rotation.from_euler('x', 45, degrees=True).as_matrix()
    translation = np.random.rand(3)
    G = Frame()
    A = Frame.orient_wrt(G, rotation=R, translation=translation)

    assert np.all(np.isclose(A.as_mat4(), mat4_from_translation_rotation(translation, R)))


def test_frame_equality():
    G = Frame()
    A = Frame()
    assert G == A
    R = Rotation.from_euler('x', 45, degrees=True).as_matrix()
    translation = np.random.rand(3)
    C = Frame.orient_wrt(G, translation=translation, rotation = R)
    D = Frame.orient_wrt(G, translation=translation, rotation = R)

    assert C == D
    assert C.is_approx(D)

