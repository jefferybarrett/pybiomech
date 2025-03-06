from pybiomech.experiment.invitro import stiffness_breakpoint_analysis, piecewise_linear
import numpy as np


def generate_piecewise_data(x, breakpoints, slopes, noise_level=0.0):
    """
    Generates piecewise linear data with optional noise.

    Parameters
    ----------
    x : ndarray
        1D array of x-values (independent variable).
    breakpoints : list
        List of x-values where slope changes.
    slopes : list
        List of slopes for each segment (length must be len(breakpoints) + 1).
    noise_level : float, optional
        Standard deviation of Gaussian noise added to x and y (default 0.0).

    Returns
    -------
    x_noisy : ndarray
        Noisy x-values.
    y_noisy : ndarray
        Noisy y-values generated from the piecewise function.
    """
    assert len(slopes) == len(breakpoints) + 1, "Number of slopes must be one more than breakpoints"

    y = piecewise_linear(x, breakpoints, slopes, 0.0)

    # Add Gaussian noise
    x_noisy = x + np.random.normal(scale=noise_level, size=x.shape)
    y_noisy = y + np.random.normal(scale=noise_level, size=y.shape)

    return x_noisy, y_noisy

def test_breakpoint_slopes():
    """Test breakpoint_slopes with a controlled piecewise linear function."""

    # Define a known piecewise function with breakpoints
    x = np.linspace(0, 10, 200)  # Smooth range of x-values
    true_breakpoints = [3, 7]  # Known x-values where slopes change
    true_slopes = [1, -0.5, 2]  # Slopes before and after breakpoints

    # Generate noisy experimental data
    np.random.seed(42)  # Fix seed for reproducibility
    x_noisy, y_noisy = generate_piecewise_data(x, true_breakpoints, true_slopes, noise_level=0.2)

    # Run breakpoint analysis
    b0, estimated_breakpoints, estimated_slopes = stiffness_breakpoint_analysis(x_noisy, y_noisy, n=2, frac=0.1)

    # Test breakpoints are roughly recovered (within ±0.5 of true values)
    np.testing.assert_allclose(estimated_breakpoints, true_breakpoints, atol=0.5)

    # Test slopes are roughly correct (within ±0.5 of true slopes)
    np.testing.assert_allclose(estimated_slopes, true_slopes, atol=0.5)

    print("Test passed: Breakpoints and slopes are correctly estimated within expected tolerance!")

# Run the test if executed as a script
if __name__ == "__main__":
    test_breakpoint_slopes()