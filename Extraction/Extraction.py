import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft
from transfer_functions import *

c = 299792458  # Speed of light in m/s

# Create extractor class
class Extractor:
    def __init__(self, reference: np.ndarray, sample: np.ndarray, thickness: float):

        ### Preprocess time domain data and extract frequency range when the class is first called

        # define physical constants
        self.Length = thickness  # Ensure it is in meters

        # Extract the signals and time data
        self.signal_ref = reference[:, 1]
        self.signal_sample = sample[:, 1]

        self.time_ref = reference[:, 0]
        self.time_sample = sample[:, 0]

        # Calculate offset and padding
        offset = self.time_sample[0] - self.time_ref[0]
        time_step = self.time_ref[1] - self.time_ref[0]
        n_padding = int(offset / time_step)

        # Adjust time and signal arrays by padding
        self.time_ref = np.concatenate([self.time_ref, np.linspace(self.time_ref[-1], self.time_sample[-1], n_padding)])
        self.time_sample = np.concatenate([np.linspace(self.time_ref[0], self.time_sample[0], n_padding), self.time_sample])
        self.signal_ref = np.concatenate([self.signal_ref, np.zeros(n_padding)])
        self.signal_sample = np.concatenate([np.zeros(n_padding), self.signal_sample])

        # Time and frequency domain parameters
        T = time_step
        Fs = 1 / T
        L = len(self.signal_ref)
        t = np.arange(0, L) * T
        self.f = Fs / L * np.arange(0, L)  # Frequency values



    ###--------------------------------------------------------------------------------------------------------
    # Plots and returns time domain data

    def plot_time_domain(self):
        # Plot the signals
        plt.figure(figsize=(8,4), dpi=150)
        plt.plot(self.time_ref, self.signal_ref, label="Reference Signal")
        plt.plot(self.time_sample, self.signal_sample, label="Sample Signal")
        plt.title('Signals in time domain')
        plt.xlabel('Time [ps]')
        plt.ylabel('Signal [nA]')
        plt.legend()
        plt.show()



    def get_processed_data(self):
        # Return arrays with data needed for later steps
        return self.signal_ref, self.time_ref, self.signal_sample, self.time_sample, self.f

    ###--------------------------------------------------------------------------------------------------------
    # Handles frequency domain data 

    def fft_signals(self, interpolation: int = 2**12):
        self.interpolation = interpolation
        # Compute FFTs of both signals with interpolation
        fft_ref = fft(self.signal_ref, interpolation)
        fft_sample = fft(self.signal_sample, interpolation)

        # Calculate amplitude and phase for both signals
        self.A_signal_ref = np.abs(fft_ref)
        self.ph_signal_ref = np.unwrap(np.angle(fft_ref))

        self.A_signal_sample = np.abs(fft_sample)
        self.ph_signal_sample = np.unwrap(np.angle(fft_sample))

        # Adjust frequency array for all possible values from fft
        self.f_interp = np.linspace(self.f[0], self.f[-1], interpolation)



    def get_fft_data(self):
        # Return amplitude, phase, and frequency data
        return self.f_interp, self.A_signal_ref, self.A_signal_sample, self.ph_signal_ref, self.ph_signal_sample



    def plot_frequency_domain(self):
        # Create a tiled layout for plotting amplitude and phase
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot amplitude spectrum
        axs[0].plot(self.f_interp, self.A_signal_ref, label='Reference Amplitude')
        axs[0].plot(self.f_interp, self.A_signal_sample, label='Sample Amplitude')
        axs[0].set_xlim([0, 4])
        axs[0].set_title('Fourier transform of Signals')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()

        # Plot phase spectrum
        axs[1].plot(self.f_interp, self.ph_signal_ref, label='Reference Phase')
        axs[1].plot(self.f_interp, self.ph_signal_sample, label='Sample Phase')
        axs[1].set_xlim([0, 4])
        axs[1].set_xlabel('Frequency [THz]')
        axs[1].set_ylabel('Phase [Radians]')
        axs[1].legend()

        # Display the plots
        plt.tight_layout()
        plt.show()

    ###--------------------------------------------------------------------------------------------------------
    # fitting method for the refractive index
    def calculate_refractive_index(self, n_0: complex):

        """Calculate refractive index using the Newton-Raphson method."""
        
        # define experimental transfer function
        H_exp_general = (self.A_signal_sample * np.exp(1j * self.ph_signal_sample)) / (self.A_signal_ref * np.exp(1j * self.ph_signal_ref))
        
        #define components of experimental transfer function
        self.A_exp = np.abs(H_exp_general)
        self.ph_exp = np.unwrap(np.angle(H_exp_general))

        # Initialize extracted array to be complex and the correct size
        self.n_extracted = np.zeros(self.interpolation, dtype=complex)
        

        # Iterate over frequencies
        w = 2 * np.pi * self.f_interp * 1e12  # Angular frequency in radians/sec
        for f_index in range(self.interpolation):
            n_next = n_0  # Reset for each frequency
            for _ in range(10):  # Arbitrary number of iterations for Newton-Raphson
                H_th = H_th_function(n_next, w)
                A_th = np.abs(H_th)
                ph_th = np.unwrap(np.angle(H_th))

                # Function to optimize
                fun = np.log(A_th[f_index]) - np.log(self.A_exp[f_index]) + 1j * ph_th[f_index] - 1j * self.ph_exp[f_index]
                fun_prime = H_prime_function(n_next, w[f_index])

                # Update refractive index using Newton-Raphson
                n_next = n_next - fun / fun_prime

            # Store extracted refractive index
            self.n_extracted[f_index] = n_next

    def get_refractive_index_data(self):
        """Return the real and imaginary parts of the extracted refractive index."""
        return np.real(self.n_extracted), np.imag(self.n_extracted)

    def plot_refractive_index(self):
        """Plot the real and imaginary parts of the refractive index."""
        fig, axs = plt.subplots(2, 1, figsize=(12, 6))

        # Plot real part of refractive index
        axs[0].plot(self.f_interp, np.real(self.n_extracted))
        axs[0].set_xlim([0, 4])
        #axs[0].set_ylim([3.45, 3.47])
        axs[0].set_xlabel("Frequency (THz)")
        axs[0].set_ylabel("Real refractive index n")

        # Plot imaginary part of refractive index (extinction coefficient)
        axs[1].plot(self.f_interp, np.imag(self.n_extracted))
        axs[1].set_xlim([0, 4])
        #axs[1].set_ylim([-0.5, 0.5])
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Extinction coefficient k")

        plt.tight_layout()
        plt.show()







###--------------------------------------------------------------------------------------------------------
# Test the functionality
if __name__ == "__main__":
    ref_tab = pd.read_csv("data/ref.pulse.csv").to_numpy()
    sample_tab = pd.read_csv("data/Si.pulse.csv").to_numpy()

    extractor = Extractor(ref_tab, sample_tab, thickness=3*1e-3)

    print(f'length of output from get_data: {len(extractor.get_processed_data())}')
    extractor.plot_time_domain()
    print()
    print('Time domain plotted successfully')

    extractor.fft_signals()
    print(f'length of output from get_fft_data: {len(extractor.get_fft_data())}')
    print()
    extractor.plot_frequency_domain()
    print('Frequency domain plotted successfully')
    print()

    extractor.calculate_refractive_index(n_0=3.7 + 0.1j)
    extractor.plot_refractive_index()
    print('Successfully extracted and plotted refractive index of sample')
