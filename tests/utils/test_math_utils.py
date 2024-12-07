import pytest
import numpy as np
from pybiomech.utils.math_utils import *


def test_xprod_mat():
    for _ in range(100):
        u = (100 * (np.random.rand(3) - 0.5)).astype(int)
        v = 100 * (np.random.rand(3) - 0.5).astype(int)

        assert np.all(np.cross(u,v) == xprod_mat(u) @ v)


def test_rot():
    pass


