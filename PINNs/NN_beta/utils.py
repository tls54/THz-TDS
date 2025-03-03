## Utilities functions for NN code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def plot_histogram(data, title="Histogram", xlabel="Value", ylabel="Frequency", color="blue"):
    """
    Plots a histogram for the given data with optimal bin sizes determined by the Freedman-Diaconis rule.

    Parameters:
        data (array-like): The input data for the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Color of the histogram bars.
    """
    # Convert data to a numpy array for processing
    data = np.asarray(data)
    
    # Calculate Freedman-Diaconis bin width
    q25, q75 = np.percentile(data, [25, 75])  # Calculate Q1 and Q3
    iqr = q75 - q25  # Interquartile range
    bin_width = 2 * iqr / (len(data) ** (1/3))  # Freedman-Diaconis formula

    # If bin_width is too small, default to sqrt(n) bins
    if bin_width == 0:
        num_bins = int(np.sqrt(len(data)))
    else:
        num_bins = int(np.ceil((data.max() - data.min()) / bin_width))

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=num_bins, color=color, alpha=0.75, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

### function to add gaussian noise to generic signal

def add_gaussian_noise(signal, mean=0, stddev=1):
    """
    Adds Gaussian noise to a given signal.
    
    Parameters:
        signal (list or np.array): The input signal to which noise will be 
added.
        mean (float): Mean of the Gaussian distribution. Default is 0.
        stddev (float): Standard deviation of the Gaussian distribution. 
Default is 1.
        
    Returns:
        np.array: Noisy signal with the same shape as the input signal.
    """
    # Convert the input to a NumPy array for easier manipulation
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, size=signal.shape)
    
    # Add noise to the signal
    noisy_signal = signal + noise
    
    return noisy_signal