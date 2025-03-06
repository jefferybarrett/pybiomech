from pybiomech.experiment.utils.breakpoint import piecewise_linear, _objective_function, breakpoint_analysis, _max_orthogonal_distance_segment
import numpy as np



def test_piecewise_linear():
    # Test Case 1: Simple linear function (no breakpoints)
    x = np.array([0, 1, 2, 3, 4, 5])
    x_b = []  # No breakpoints
    M = [2]   # Single slope
    b = 1     # y-intercept
    expected = 2 * x + 1
    np.testing.assert_allclose(piecewise_linear(x, x_b, M, b), expected, rtol=1e-5)

    # Test Case 2: One breakpoint at x=2
    x = np.array([0, 1, 2, 3, 4, 5])
    x_b = [2]
    M = [1, -0.5]  # Slope 1 before x=2, then -0.5
    b = 0
    expected = np.array([0, 1, 2, 1.5, 1, 0.5])  # Manually computed expected values
    np.testing.assert_allclose(piecewise_linear(x, x_b, M, b), expected, rtol=1e-5)

    # Test Case 3: Two breakpoints at x=2 and x=4
    x_b = [2, 4]
    M = [1, -0.5, 2]  # Slopes before and after each breakpoint
    b = 0
    expected = np.array([0, 1, 2, 1.5, 1, 3])  # Manually verified
    np.testing.assert_allclose(piecewise_linear(x, x_b, M, b), expected, rtol=1e-5)

    # Test Case 4: Edge Case - x exactly at a breakpoint
    x_edge = np.array([2, 4])  # x at breakpoints
    expected_edge = np.array([2, 1])  # Should match function continuity
    np.testing.assert_allclose(piecewise_linear(x_edge, x_b, M, b), expected_edge, rtol=1e-5)

    # Test Case 5: Negative slopes
    x_b = [1, 3]
    M = [-2, 0.5, -1]  # Different slopes
    b = 5
    expected = np.array([5, 3, 3.5, 4.0, 3.0, 2.0])  # Computed manually
    np.testing.assert_allclose(piecewise_linear(x, x_b, M, b), expected, rtol=1e-5)

    print("All piecewise_linear tests passed!")


def test_objective_function():
    """Test the objective_function against known values and edge cases."""
    
    # Test case 1: Simple linear model (no breakpoints)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])  # Perfect line y = 2x
    p = np.array([0, 2])  # intercept=0, slope=2, no breakpoints
    
    # With perfect fit, error should be zero
    assert _objective_function(p, x, y, n=0) == 0
    
    # Test case 2: Simple linear model with error
    y_with_error = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    error = np.sum((y_with_error - (2*x))**2)
    assert np.isclose(_objective_function(p, x, y_with_error, n=0), error)
    
    # Test case 3: Piecewise linear model with one breakpoint
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 1.5, 1, 0.5])  # Slope 1 before x=2, then -0.5
    p = np.array([0, 2, 1, -0.5])  # intercept=0, breakpoint=2, slopes=[1, -0.5]
    
    # With perfect fit, error should be zero
    assert np.isclose(_objective_function(p, x, y, n=1), 0)
    
    # Test case 4: With weights
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 9, 11])  # Not a perfect fit
    y_model = np.array([2, 4, 6, 8, 10])  # Perfect line y = 2x
    p = np.array([0, 2])  # intercept=0, slope=2, no breakpoints
    
    # Calculate expected error manually
    residuals = y - y_model
    expected_error = np.sum(residuals**2)  # Unweighted
    assert np.isclose(_objective_function(p, x, y, n=0), expected_error)
    
    # With weights that emphasize later points
    weights = np.array([1, 1, 1, 2, 2])
    expected_weighted_error = np.sum((weights * residuals)**2)
    assert np.isclose(_objective_function(p, x, y, n=0, w=weights), expected_weighted_error)
    
    # Test case 5: Piecewise model with multiple breakpoints
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Slopes: 1 (x≤2), 0.5 (2<x≤6), -1 (x>6)
    y = np.array([0, 1, 2, 2.5, 3, 3.5, 4, 3, 2, 1, 0])
    p = np.array([0, 2, 6, 1, 0.5, -1])  # intercept=0, breakpoints=[2,6], slopes=[1,0.5,-1]
    
    # With perfect fit, error should be zero
    assert np.isclose(_objective_function(p, x, y, n=2), 0)
    
    # Test case 6: Edge case with empty inputs
    x_empty = np.array([])
    y_empty = np.array([])
    p_empty = np.array([0, 1])  # Some arbitrary parameters
    assert _objective_function(p_empty, x_empty, y_empty, n=0) == 0  # Empty sum is zero    


def test_max_orthogonal_distance_segment():
    """Test the max_orthogonal_distance_segment function against various scenarios."""
    
    # Test case 1: Simple triangle - the apex should have the max distance
    x = np.array([0, 5, 10])
    y = np.array([0, 5, 0])
    
    # The secant connects (0,0) and (10,0), and the middle point (5,5) has distance 5
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x, y)
    assert x_max == 5
    assert y_max == 5
    assert np.isclose(dist_max, 5.0)
    
    # Test case 2: Straight line - all points should have distance 0 (or very close)
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 3, 4, 5])
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x, y)
    assert np.isclose(dist_max, 0.0, atol=1e-10)
    
    # Test case 3: Empty inputs
    x_empty = np.array([])
    y_empty = np.array([])
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x_empty, y_empty)
    assert dist_max == -np.inf
    
    # Test case 4: Single point
    x_single = np.array([3])
    y_single = np.array([4])
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x_single, y_single)
    assert x_max == 3
    assert y_max == 4
    assert dist_max == 0.0
    
    # Test case 5: Vertical line (edge case where x[-1] = x[0])
    x = np.array([5, 5, 5, 5, 5])
    y = np.array([0, 1, 3, 7, 10])
    # The secant is x = 5, so the distance is |x_i - 5|
    # Since all x values are 5, the max distance should be 0
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x, y)
    assert np.isclose(dist_max, 0.0)
    
    # Test case 6: Horizontal line with one outlier
    x = np.array([0, 2, 5, 7, 10])
    y = np.array([0, 0, 3, 0, 0])
    # The point (5,3) should have max distance 3
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x, y)
    assert x_max == 5
    assert y_max == 3
    assert np.isclose(dist_max, 3.0)
    
    # Test case 7: More complex shape
    x = np.array([0, 2, 4, 6, 8, 10])
    y = np.array([0, 4, 2, 5, 1, 0])
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x, y)
    
    # Calculate expected distance manually
    # Line equation between (0,0) and (10,0) is y = 0
    # So the orthogonal distance for each point is just |y|
    expected_index = np.argmax(np.abs(y))
    expected_x = x[expected_index]
    expected_y = y[expected_index]
    expected_dist = abs(expected_y)
    
    assert x_max == expected_x
    assert y_max == expected_y
    assert np.isclose(dist_max, expected_dist)

    # Test case 8: Lists instead of numpy arrays
    x_list = [0, 5, 10]
    y_list = [0, 5, 0]
    x_max, y_max, dist_max = _max_orthogonal_distance_segment(x_list, y_list)
    assert x_max == 5
    assert y_max == 5
    assert np.isclose(dist_max, 5.0)


def test_breakpoint_analysis():
    """Tests breakpoint_analysis with various scenarios."""

    # Test Case 1: Simple linear function (no real breakpoints)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 5  # Straight line (no actual breakpoints)
    
    b0, x_b, m_v = breakpoint_analysis(x, y, n=2, show_plots=False, verbose=False)

    # Expect slopes to be approximately 2 everywhere
    np.testing.assert_allclose(m_v, [2, 2, 2], rtol=1e-2)
    assert len(x_b) == 2  # Should have two breakpoints
    assert np.min(x) <= min(x_b) <= max(x_b) <= np.max(x)  # Breakpoints must be within range

    # Test Case 2: A piecewise function with known breakpoints
    x = np.linspace(0, 10, 100)
    y = piecewise_linear(x, [4.0, 8.0], [-1.0, 2.0, 3.0], 10.0)
    
    b0, x_b, m_v = breakpoint_analysis(x, y, n=2, show_plots=False, verbose=False)

    # Breakpoints should be close to x=3 and x=7
    np.testing.assert_allclose(x_b, [4.0, 8.0], atol=0.5)  
    np.testing.assert_allclose(m_v, [-1.0, 2.0, 3.0], atol=0.5)  # Slopes should match the segments

    # Test Case 3: Edge Case - Zero Breakpoints (should just return a single slope)
    b0, x_b, m_v = breakpoint_analysis(x, y, n=0, show_plots=False, verbose=False)

    assert len(x_b) == 0  # No breakpoints
    assert len(m_v) == 1  # One slope only

    # Test Case 4: Noisy Data - Ensures robustness
    np.random.seed(42)  # Fix seed for reproducibility
    noise = np.random.normal(scale=0.2, size=len(y))
    y_noisy = y + noise  # Add noise to the data

    b0, x_b, m_v = breakpoint_analysis(x, y_noisy, n=2, show_plots=False, verbose=False)

    # Breakpoints should still be reasonably close to original locations
    np.testing.assert_allclose(x_b, [3, 7], atol=1.0)

    print("All breakpoint_analysis tests passed!")


# Run the test if executed as a script
if __name__ == "__main__":
    test_piecewise_linear()
