import numpy as np
from numpy.fft import fft



def fft_signals(signal_ref, signal_sample, interpolation: int, f_interp, unwrapping_regression_range):
        # Compute FFTs of both signals with interpolation
        fft_ref = fft(signal_ref, interpolation)
        fft_sample = fft(signal_sample, interpolation)

        # Calculate amplitude and phase for both signals
        A_signal_ref = np.abs(fft_ref)
        ph_signal_ref = np.unwrap(np.angle(fft_ref))

        A_signal_sample = np.abs(fft_sample)
        ph_signal_sample = np.unwrap(np.angle(fft_sample))

        # fit a linear model to the phase in a given range      
        finds = np.arange(*unwrapping_regression_range)
        coef_r = np.polyfit(f_interp[finds], ph_signal_ref[finds], 1)
        coef_s = np.polyfit(f_interp[finds], ph_signal_sample[finds], 1)

        # remove the offset
        ph_signal_ref -= coef_r[1] 
        ph_signal_sample -= coef_s[1] 

        # mean squared difference between the fit and data (sanity check)
        mse_r = np.mean(np.square(ph_signal_ref-f_interp*coef_r[0])[finds])
        mse_s = np.mean(np.square(ph_signal_sample-f_interp*coef_s[0])[finds])

        print("phase offset fit frequency range: ", f_interp[unwrapping_regression_range])
        print("slopes for reference  and sample: ", coef_r[0], coef_s[0])
        print("mean squared error for ref and sample: ", mse_r, mse_s)
        print("(should be ~< 1)")

        return A_signal_ref, ph_signal_ref, A_signal_sample, ph_signal_sample