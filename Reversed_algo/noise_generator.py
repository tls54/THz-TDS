## Utility files for data generation package
import numpy as np
import time


def noise_generator(signal: np.array, noise_bounds: list, noise_profile: str = "uniform"):
    
    """
    Generate a noisy version of the input signal.

    Parameters:
    ------------
        signal (np.array): The original signal to be modified.
        noise_bounds (list): A list containing two values representing the minimum and maximum bounds for noise addition. 
        For example: [0, 10] would add noise between 0 and 10 times the standard deviation of the input signal.
        noise_profile (str): The distribution profile for noise to be added. Can be one of 'uniform', 'gaussian' or 'normal'.
        Default is 'uniform'.

    Returns:
    --------
        np.array: The noisy version of the original signal.

    Raises:
    -------
        TypeError: If noise_bounds is not a list, or if noise_profile is not a string.

        ValueError: If noise_bounds does not contain exactly two values, 
        or if min_noise is greater than max_noise,
        or if an invalid value is passed for noise_profile.

    Notes:
    ------
        This function uses NumPy's random number generator to add noise to 
        the input signal. 
        The type of distribution (uniform, gaussian, normal) and its bounds are determined by the user-provided noise_bounds and noise_profile.
    """



    # check metadata of inputs
    if not isinstance(noise_bounds, list):
        raise TypeError("Expected input to be a list.")
    if len(noise_bounds) != 2:
        raise ValueError("Expected a list of length 2.")
    
    if not isinstance(noise_profile, str):
        raise TypeError(f"Expected str but got {type(noise_profile)}.")

    # extract min and max noise to check validity
    min_noise, max_noise = noise_bounds
    # handles comparison of complex noise parameters 
    if min_noise.real > max_noise.real or min_noise.imag > max_noise.imag:
        raise ValueError(f"Minimum noise bound ({min_noise}) cannot have larger real or imaginary parts than maximum noise bound ({max_noise}).")
    
    # define random seed of random number gen
    seed = int(time.time() * 1000)  # ms precision
    rng = np.random.default_rng(seed)

    data_points = len(signal)

    if noise_profile == "uniform":  # noise distribution is uniform
        if isinstance(min_noise, complex) or isinstance(max_noise, complex):
            # Handle complex noise: Generate uniform noise separately for real and imaginary parts
            real_part = rng.uniform(low=min_noise.real, high=max_noise.real, size=data_points)
            imag_part = rng.uniform(low=min_noise.imag, high=max_noise.imag, size=data_points)
            noise_adjustments = real_part + 1j * imag_part
        else:
            # Handle real noise
            noise_adjustments = rng.uniform(low=min_noise, high=max_noise, size=data_points)

    elif noise_profile in {"gaussian", "normal"}:  # noise follows normal distribution
        if isinstance(min_noise, complex) or isinstance(max_noise, complex):
            # Handle complex noise: Generate Gaussian noise separately for real and imaginary parts
            mean_real = (min_noise.real + max_noise.real) / 2  # Midpoint of real parts
            std_dev_real = (max_noise.real - min_noise.real) / 2  # Half the range of real parts
            real_part = rng.normal(loc=mean_real, scale=std_dev_real, size=data_points)
            
            mean_imag = (min_noise.imag + max_noise.imag) / 2  # Midpoint of imaginary parts
            std_dev_imag = (max_noise.imag - min_noise.imag) / 2  # Half the range of imaginary parts
            imag_part = rng.normal(loc=mean_imag, scale=std_dev_imag, size=data_points)
            
            noise_adjustments = real_part + 1j * imag_part
        else:
            # Handle real noise
            mean = (min_noise + max_noise) / 2  # Midpoint of the bounds
            std_dev = (max_noise - min_noise) / 2  # Half the range of the bounds
            noise_adjustments = rng.normal(loc=mean, scale=std_dev, size=data_points)

    else:  # If the profile is not valid, raise an error
        raise ValueError(f"Invalid noise_profile '{noise_profile}'. Must be one of ['uniform', 'gaussian', 'normal'].")

    # Apply noise to the original signal
    noisy_signal = signal + noise_adjustments

    return noisy_signal
