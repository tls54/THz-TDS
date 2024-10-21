import numpy as np
from numpy.fft import fft



def fft_signals(signal_ref, signal_sample, interpolation: int):
        interpolation = interpolation
        # Compute FFTs of both signals with interpolation
        fft_ref = fft(signal_ref, interpolation)
        fft_sample = fft(signal_sample, interpolation)

        # Calculate amplitude and phase for both signals
        A_signal_ref = np.abs(fft_ref)
        ph_signal_ref = np.unwrap(np.angle(fft_ref))

        A_signal_sample = np.abs(fft_sample)
        ph_signal_sample = np.unwrap(np.angle(fft_sample))


        return A_signal_ref, ph_signal_ref, A_signal_sample, ph_signal_sample