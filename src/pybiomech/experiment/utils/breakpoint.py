import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def polynomial_method(x, y, order=6):
    """
    Identifies breakpoints in a curve by finding peaks in the second derivative 
    of a high-order polynomial fit.

    This method fits a polynomial of the specified order to `(x, y)` data, 
    computes its second derivative, and finds its peaks by solving for 
    roots of the third derivative. Points where the fourth derivative 
    is negative (indicating maxima) are retained as breakpoints.

    This method is adapted from:
    Beach, T. A., Parkinson, R. J., Stothart, J. P., & Callaghan, J. P. (2005). 
    Effects of prolonged sitting on the passive flexion stiffness of the 
    in vivo lumbar spine. The Spine Journal, 5(2), 145-154.

    Parameters
    ----------
    x : array_like
        1D array of x-values (independent variable).
    y : array_like
        1D array of y-values (dependent variable).
    order : int, optional
        Order of the polynomial to fit. Default is 6.

    Returns
    -------
    roots : ndarray
        Sorted array of breakpoint locations (x-values) where 
        maximal change occurs.

    Notes
    -----
    - This approach assumes the input data follows a smooth trend that 
      can be well-approximated by a polynomial of the given order.
    - Selecting too high an order may lead to overfitting, while too low 
      may miss important breakpoints.
    """
    # Fit a high-order polynomial to the x-y data
    p = np.polyfit(x, y, order)
    
    # Compute the second derivative
    ddp = np.polyder(p, m=2)
    
    # Find roots of the third derivative (where second derivative peaks)
    d3p = np.polyder(ddp, m=1)
    roots = np.roots(d3p)
    
    # Compute the fourth derivative
    d4p = np.polyder(d3p, m=1)
    fourth_derivative_values = np.polyval(d4p, roots)
    
    # Keep only the roots where the fourth derivative is negative (maxima)
    roots = np.sort([r for r, v in zip(roots, fourth_derivative_values) if v < 0])
    
    return roots


def piecewise_linear(x, x_b, M, b):
    """
    Evaluates a piecewise linear function given breakpoints, segment slopes, and an initial y-intercept.

    This function defines a piecewise linear relationship where the slopes (`M`) change 
    at specified breakpoints (`x_b`). The function ensures **continuity** by computing 
    segment-specific intercepts based on the given initial y-intercept (`b`).

    Parameters
    ----------
    x : array_like
        1D array of x-values at which the function should be evaluated.
    x_b : array_like
        1D array of x-values where slope changes (breakpoints). Must be sorted in ascending order.
    M : array_like
        1D array of slopes for each segment. Must have length `len(x_b) + 1`.
    b : float
        The y-intercept of the segment that contains `x=0`.

    Returns
    -------
    result : ndarray
        1D array of y-values corresponding to `x` based on the piecewise linear function.

    Notes
    -----
    - The function **ensures continuity** by computing the intercepts for each segment.
    - If `x_b` is empty, the function returns a simple linear function: `M[0] * x + b`.
    - The function **handles cases where `x=0` is inside any segment** and adjusts intercepts accordingly.
    - It efficiently determines the correct segment for each `x` value and applies the appropriate slope and intercept.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> x_b = [2, 4]  # Breakpoints at x = 2 and x = 4
    >>> M = [1, -0.5, 2]  # Slopes for each segment
    >>> b = 0  # Initial y-intercept
    >>> y = piecewise_linear(x, x_b, M, b)
    >>> print(y)
    [ 0.   1.   2.   1.5  1.   4. ]
    
    # Example with no breakpoints (pure linear function)
    >>> x_b = []  # No breakpoints
    >>> M = [2]   # Single slope
    >>> b = 1
    >>> y = piecewise_linear(x, x_b, M, b)
    >>> print(y)
    [ 1.  3.  5.  7.  9. 11.]
    """
    x = np.asarray(x)
    x_b = np.array(x_b) if not isinstance(x_b, np.ndarray) else x_b
    M = np.array(M) if not isinstance(M, np.ndarray) else M
    
    # Create result array
    result = np.zeros_like(x, dtype=float)
    
    # Special case: no breakpoints
    if len(x_b) == 0:
        return M[0] * x + b
    
    # First, calculate the function value at x=0 using the provided y-intercept
    f_at_zero = b
    
    # We need to work backwards to get the intercept for the first segment
    # Find which segment contains x=0
    segment_for_zero = 0
    for i, breakpoint in enumerate(x_b):
        if 0 > breakpoint:
            segment_for_zero += 1
    
    # Calculate the intercept for each segment
    intercepts = np.zeros(len(M))
    
    # For the segment containing x=0, use the given y-intercept
    intercepts[segment_for_zero] = f_at_zero
    
    # Calculate backwards for segments before x=0
    for i in range(segment_for_zero, 0, -1):
        # Value at previous breakpoint
        prev_value = M[i] * x_b[i-1] + intercepts[i]
        # Intercept for previous segment
        intercepts[i-1] = prev_value - M[i-1] * x_b[i-1]
    
    # Calculate forwards for segments after x=0
    for i in range(segment_for_zero, len(M)-1):
        # Value at current breakpoint
        curr_value = M[i] * x_b[i] + intercepts[i]
        # Intercept for next segment
        intercepts[i+1] = curr_value - M[i+1] * x_b[i]
    
    # Evaluate the function for each segment
    # First segment
    mask = x <= x_b[0]
    result[mask] = M[0] * x[mask] + intercepts[0]
    
    # Middle segments
    for i in range(len(x_b)-1):
        mask = (x > x_b[i]) & (x <= x_b[i+1])
        result[mask] = M[i+1] * x[mask] + intercepts[i+1]
    
    # Last segment
    mask = x > x_b[-1]
    result[mask] = M[-1] * x[mask] + intercepts[-1]
    
    return result


def _objective_function(p, x, y, n=2, w=None):
    """
    Calculate the weighted sum of squared errors between actual values and 
    a piecewise linear model prediction.
    
    Parameters
    ----------
    p : array_like
        Model parameters in the format [b0, x_b[0], x_b[1], ..., m_v[0], m_v[1], ...].
        b0 is the initial intercept, x_b are the breakpoints, m_v are the slopes.
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values (observations).
    n : int, optional
        Number of breakpoints in the piecewise linear model. Default is 2.
    w : array_like, optional
        Weights for each observation. If None, all observations are equally weighted.
    
    Returns
    -------
    float
        Weighted sum of squared errors between y and the piecewise linear model prediction.
        
    Notes
    -----
    This function is designed to be used as the objective function in an optimization
    routine for fitting a piecewise linear model to data.
    """
    # Default weights to ones if not provided
    w = np.ones_like(x) if w is None else w
    
    # Extract parameters
    b0 = p[0]           # Initial intercept
    x_b = p[1:n+1]      # Breakpoints
    m_v = p[n+1:]       # Slopes for each segment
    
    # Calculate model predictions
    y_tilde = piecewise_linear(x, x_b, m_v, b0)
    
    # Return weighted sum of squared errors
    return np.sum((w * (y - y_tilde))**2)


def _max_orthogonal_distance_segment(x, y):
    """
    Find the point in a set that has the maximum orthogonal distance to the secant line
    connecting the first and last points.
    
    This is commonly used in mesh simplification algorithms like the Douglas-Peucker algorithm
    for reducing the number of points in a polyline while preserving its shape.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of the points.
    y : array_like
        Y-coordinates of the points.
        
    Returns
    -------
    float
        X-coordinate of the point with maximum orthogonal distance.
    float
        Y-coordinate of the point with maximum orthogonal distance.
    float
        Maximum orthogonal distance value. Returns -np.inf if input is empty.
        
    Notes
    -----
    The orthogonal distance from a point (x_i, y_i) to the line ax + by + c = 0
    is calculated as |ax_i + by_i + c| / sqrt(a² + b²).
    """
    # Check for empty inputs
    if len(x) == 0:
        return x, y, -np.inf
    
    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)
    
    # If only one point, return that point with zero distance
    if len(x) == 1:
        return x[0], y[0], 0.0
    
    # Calculate the secant line parameters: ax + by + c = 0
    # We know b = -1, so we're solving for a and c
    # Point-slope form: (y - y₁) = m(x - x₁) where m = (y₂ - y₁)/(x₂ - x₁)
    # Rearranged to ax + by + c = 0 form where a = m, b = -1, c = y₁ - m*x₁
    
    # Calculate slope (checking for vertical line)
    if x[-1] == x[0]:
        # Vertical line case: x = constant (ax + by + c = 0 becomes x - x₀ = 0)
        # For vertical lines, distance is |x_i - x₀|
        distances = np.abs(x - x[0])
        i = np.argmax(distances)
        return x[i], y[i], distances[i]
    
    # Standard case: calculate line parameters
    a = (y[-1] - y[0]) / (x[-1] - x[0])
    b = -1.0
    c = y[0] - a * x[0]
    
    # Calculate orthogonal distances using vectorized operations
    # Distance = |ax + by + c| / sqrt(a² + b²)
    numerator = np.abs(a * x + b * y + c)
    denominator = np.sqrt(a**2 + b**2)
    distances = numerator / denominator
    
    # Find the point with maximum distance
    i = np.argmax(distances)
    
    return x[i], y[i], distances[i]


def _max_orthogonal_distance(x, y, xb, yb):
    """
    Find the point with maximum orthogonal distance to a piecewise linear curve.
    
    This function divides the x-range into segments defined by breakpoints and
    finds the point with the overall maximum orthogonal distance to the secant lines
    across all segments.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of the candidate points.
    y : array_like
        Y-coordinates of the candidate points.
    xb : array_like
        X-coordinates of the breakpoints defining the piecewise linear curve.
    yb : array_like
        Y-coordinates of the breakpoints defining the piecewise linear curve.
        
    Returns
    -------
    float or None
        X-coordinate of the point with maximum orthogonal distance.
        Returns None if no valid point is found.
    float or None
        Y-coordinate of the point with maximum orthogonal distance.
        Returns None if no valid point is found.
        
    Notes
    -----
    This function is commonly used in polyline simplification algorithms like
    Douglas-Peucker to identify the most significant points to retain. It operates 
    by finding the maximum orthogonal distance in each segment of a piecewise linear
    curve and then selecting the overall maximum.
    
    The function relies on max_orthogonal_distance_segment to calculate the maximum
    distance within each segment.
    
    See Also
    --------
    max_orthogonal_distance_segment : Function to find the maximum orthogonal distance
                                      within a single segment.
    """
    xc, yc, dist = None, None, -np.inf
    for (xl, xu) in zip(xb[:-1], xb[1:]):
        x_restricted = x[(x > xl) & (x < xu)]
        y_restricted = y[(x > xl) & (x < xu)]
        xi, yi, di = _max_orthogonal_distance_segment(x_restricted, y_restricted)
        
        if (di > dist):
            xc, yc, dist = xi, yi, di

    return xc, yc


def _breakpoint_guess(x, y, n = 2, x0 = [], y0 = []):
    """
    Estimates `n` breakpoints in (x, y) data using a Douglas-Peucker-style approach.

    This function iteratively selects breakpoints by identifying the data point 
    that is farthest from the current piecewise linear approximation, similar to 
    the Douglas-Peucker algorithm but without recursive segment splitting.

    Parameters
    ----------
    x : array_like
        1D array of x-values (independent variable), assumed to be sorted.
    y : array_like
        1D array of y-values (dependent variable).
    n : int, optional
        Number of breakpoints to estimate (default is 2).
    x0 : list, optional
        List of previously identified breakpoint x-values. Defaults to empty.
    y0 : list, optional
        List of previously identified breakpoint y-values. Defaults to empty.

    Returns
    -------
    x_b : ndarray
        Array of x-coordinates for the estimated breakpoints, including endpoints.
    y_b : ndarray
        Array of y-coordinates for the estimated breakpoints, including endpoints.

    Notes
    -----
    - This method is inspired by the **Douglas-Peucker algorithm**, but instead of 
      recursively splitting line segments, it iteratively selects the point 
      with **maximum orthogonal distance** to the current approximation.
    - Ensures **breakpoints remain sorted** in increasing x-order.
    - Works well for moderate `n`, but recursion may be inefficient for very large `n`.

    Example
    -------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + 0.1 * np.random.randn(100)
    >>> x_b, y_b = _breakpoint_guess(x, y, n=3)
    >>> print(x_b)
    >>> print(y_b)
    """
    if (n == 0):
        # no breakpoints requested, AND there are no current breakpoints
        # just return the two ends of the array
        return np.array([x[0]] + x0 + [x[-1]]), np.array([y[0]] + y0 + [y[-1]])
    else:
        # use the current breakpoints to partition the current x and y
        x1, y1 = np.array([x[0]] + x0 + [x[-1]]), np.array([y[0]] + y0 + [y[-1]])
        
        # then find the point that has the maximum orthogonal distance to this approx
        xj, yj = _max_orthogonal_distance(x, y, x1, y1)
        
        x0.append(xj)
        y0.append(yj)
        if (len(x0) > 1):
            inds = np.argsort(x0)
            x0 = list(np.array(x0)[inds])
            y0 = list(np.array(y0)[inds])
        
        return _breakpoint_guess(x, y, n = n - 1, x0 = x0, y0 = y0)
        

def _initial_guess(x, y, n = 2):
    """
    Generates an initial breakpoint estimate for segmented regression.

    This function determines `n` breakpoints using the `_breakpoint_guess` function 
    and computes the corresponding piecewise linear slopes and intercept.

    Parameters
    ----------
    x : array_like
        1D array of x-values (independent variable), assumed to be sorted.
    y : array_like
        1D array of y-values (dependent variable).
    n : int, optional
        Number of breakpoints to estimate (default is 2).

    Returns
    -------
    x_b : ndarray
        Array of estimated x-coordinates for breakpoints (length `n`).
    m_v : ndarray
        Array of segment slopes (length `n + 1`).
    b0 : float
        y-intercept of the first segment.

    Notes
    -----
    - Uses `_breakpoint_guess` to select breakpoints based on the **maximum orthogonal distance** approach.
    - Computes slopes (`m_v`) for each segment using **finite differences**.
    - The first segment's y-intercept (`b0`) is extracted from `y0[0]`.
    - The estimated breakpoints `x_b` exclude the first and last data points 
      (which are used as fixed endpoints).

    Example
    -------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + 0.1 * np.random.randn(100)
    >>> x_b, m_v, b0 = _initial_guess(x, y, n=3)
    >>> print(x_b)  # Estimated breakpoints
    >>> print(m_v)  # Slopes for each segment
    >>> print(b0)   # First segment's y-intercept
    """
    # Estimate `n` breakpoints using max orthogonal distance criterion
    x0, y0 = _breakpoint_guess(x, y, n, [], [])

    # First segment's y-intercept
    b0 = y0[0]

    # Compute slopes for each segment using finite differences
    m_v = np.diff(y0) / np.diff(x0)

    # Extract breakpoints (excluding first & last point)
    x_b = x0[1:-1]

    return x_b, m_v, b0


def breakpoint_analysis(x, y, n=2, show_plots=False, verbose=False):
    """
    Performs segmented regression (breakpoint analysis) on (x, y) data using `n` breakpoints.

    This function estimates breakpoints in a dataset by fitting a piecewise linear 
    function with `n+1` segments. It first determines an initial guess for the breakpoints
    using `_initial_guess` and then optimizes the locations and slopes via 
    least-squares minimization.

    Parameters
    ----------
    x : array_like
        1D array of x-values (independent variable), assumed to be sorted.
    y : array_like
        1D array of y-values (dependent variable).
    n : int, optional
        Number of breakpoints to estimate (default is 2).
    show_plots : bool, optional
        If True, displays a plot comparing the piecewise linear fit to the data.
    verbose : bool, optional
        If True, prints optimization details to the console.

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
    - Uses `_initial_guess` to obtain a starting point for optimization.
    - The function optimizes the breakpoints and slopes using `scipy.optimize.minimize`.
    - Ensures the breakpoints remain within the valid range `[min(x), max(x)]`.
    - The function fits a **continuous** but **not necessarily smooth** piecewise linear function.

    Example
    -------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.piecewise(x, [x < 3, (x >= 3) & (x < 7), x >= 7], 
    ...                  [lambda x: 2*x + 1, lambda x: -1*x + 10, lambda x: 0.5*x + 2])
    >>> b0, x_b, m_v = breakpoint_analysis(x, y, n=2, show_plots=True, verbose=True)
    >>> print("Breakpoints:", x_b)
    >>> print("Slopes:", m_v)
    >>> print("Intercept:", b0)
    """
    # Step 1: Get an initial guess for breakpoints and slopes
    x_b, m_v, b0 = _initial_guess(x, y, n)

    # Step 2: Define optimization bounds
    b0_bnds = [(-np.inf, np.inf)]  # y-intercept is unrestricted
    x_b_bnds = [(np.min(x), np.max(x)) for _ in x_b]  # Breakpoints must remain in x-range
    m_v_bnds = [(-np.inf, np.inf) for _ in m_v]  # Slopes are unrestricted

    # Step 3: Construct parameter vector and bounds for optimization
    p0 = np.array([b0] + list(x_b) + list(m_v))  # Initial parameter guess
    p_bnds = b0_bnds + x_b_bnds + m_v_bnds  # Bounds for parameters

    # Step 4: Optimize breakpoints and slopes using least-squares minimization
    sol = minimize(_objective_function, p0, args=(x, y, n), bounds=p_bnds)

    # Step 5: Extract optimized parameters
    b0, x_b, m_v = sol.x[0], sol.x[1:n+1], sol.x[n+1:]

    # Step 6: Print optimization details if requested
    if verbose:
        print("Optimization Result:")
        print(sol)

    # Step 7: Plot results if requested
    if show_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'ro', label="Data")
        plt.plot(x, piecewise_linear(x, x_b, m_v, b0), 'k-', label="Piecewise Fit")
        plt.scatter(x_b, piecewise_linear(x_b, x_b, m_v, b0), color='black', zorder=3, label="Breakpoints")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Breakpoint Analysis - Piecewise Linear Fit")
        plt.show()

    return b0, x_b, m_v