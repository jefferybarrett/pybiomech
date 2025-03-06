import numpy as np
import statsmodels.api as sm
from pybiomech.experiment.utils.breakpoint import breakpoint_analysis, piecewise_linear


def _lowess_curve(X, Y, frac=0.1, num_points=None, **kwargs):
    """
    Applies LOWESS smoothing to experimental data and resamples it to evenly spaced points.

    Parameters
    ----------
    X : array_like
        The independent variable (e.g., time, force, displacement).
    Y : array_like
        The dependent variable (e.g., reaction force, pressure, angle).
    frac : float, optional
        The LOWESS smoothing parameter, controlling the degree of smoothing (default 0.1).
        - Lower values preserve more local detail.
        - Higher values smooth more aggressively.
    num_points : int, optional
        Number of points to resample the smoothed curve to (default is `min(256, len(X))`).
    kwargs : dict, optional
        Additional parameters for `statsmodels.nonparametric.lowess`.

    Returns
    -------
    x : ndarray
        Evenly spaced x-values for the smoothed curve.
    y : ndarray
        Interpolated y-values from the LOWESS-smoothed curve.

    Notes
    -----
    - This function **performs LOWESS smoothing** to reduce noise in experimental data.
    - The output `x` is evenly spaced to **avoid irregular step sizes**.
    - `num_points` defaults to `min(256, len(X))` to avoid excessive interpolation.
    """
    # Ensure correct argument order for LOWESS (Y first, X second)
    lowess = sm.nonparametric.lowess(Y, X, frac=frac, **kwargs)
    
    lowess_x, lowess_y = lowess[:, 0], lowess[:, 1]
    
    # Determine number of resampled points
    if num_points is None:
        num_points = min(256, len(X))  # Avoid excessive upsampling

    x = np.linspace(np.min(lowess_x), np.max(lowess_x), num_points)
    y = np.interp(x, lowess_x, lowess_y)  # Interpolate smoothed curve

    return x, y


def stiffness_breakpoint_analysis(X, Y, n=4, frac=0.1):
    """
    Performs breakpoint analysis on smoothed experimental data.

    This function applies LOWESS smoothing to `(X, Y)`, then performs 
    **segmented regression** to estimate breakpoints and segment slopes.

    Parameters
    ----------
    X : array_like
        The independent variable (e.g., time, force, displacement).
    Y : array_like
        The dependent variable (e.g., reaction force, pressure, angle).
    n : int, optional
        Number of breakpoints to estimate (default is 4).
    frac : float, optional
        The LOWESS smoothing parameter, controlling the degree of smoothing (default 0.1).

    Returns
    -------
    b0 : float
        y-intercept of the first segment.
    x_b : ndarray
        Array of optimized x-coordinates for breakpoints (length `n`).
    m_v : ndarray
        Array of segment slopes (length `n + 1`).

    Notes
    -----
    - LOWESS smoothing **reduces experimental noise** before segmentation.
    - The function estimates `n+1` segment slopes based on **piecewise linear fits**.
    - `n` should be chosen based on the expected number of breakpoints in the data.
    """
    # Apply LOWESS smoothing
    x, y = _lowess_curve(X, Y, frac=frac)

    # Perform breakpoint analysis on smoothed data
    b0, x_b, m_v = breakpoint_analysis(x, y, n=n, show_plots=False, verbose=False)

    return b0, x_b, m_v




