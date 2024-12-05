import numpy as np

__all__ = [
    "PI",
    "TAU",
    "ihat",
    "jhat",
    "khat",
    "ORIGIN",
]

PI = np.pi
TAU = 2*PI
ORIGIN = np.array([0.0, 0.0, 0.0])
ihat = np.array([1.0, 0.0, 0.0])
jhat = np.array([0.0, 1.0, 0.0])
khat = np.array([0.0, 0.0, 1.0])
