# Extractor.py Documentation

The `Extractor` class is designed for extracting material parameters using THz Time-Domain Spectroscopy (THz-TDS). It facilitates time-domain signal processing, frequency-domain transformations, and refractive index calculations.

---

## Overview

The `Extractor` class performs the following key tasks:
- Preprocessing and visualization of time-domain data.
- Frequency-domain analysis using Fast Fourier Transform (FFT).
- Calculation of refractive indices via the Newton-Raphson method.
- Plotting for data visualization.

---

## Class: `Extractor`

### Initialization

The `Extractor` class initializes with reference and sample signals, material thickness, and an optional phase unwrapping regression range.

#### Parameters
- **`signal_ref`**: Reference signal in the time domain.
- **`signal_sample`**: Sample signal in the time domain.
- **`thickness`**: Thickness of the material (in meters).
- **`dc_offset_range`**: Calculates the DC offset or the time domain signals default is set to use the first 50 values.

---

### Attributes

#### Signal and Time
- **`signal_ref`**: Reference signal in the time domain.
- **`signal_sample`**: Sample signal in the time domain.
- **`time_ref`**: Time values corresponding to the reference signal.
- **`time_sample`**: Time values corresponding to the sample signal.

#### Frequency Domain
- **`f`**: Frequency array calculated from the time domain data.
- **`f_interp`**: Interpolated frequency array after FFT.
- **`A_transfer`**: Amplitude of the transfer function in the frequency domain.
- **`ph_transfer`**: Unwrapped phase of the transfer function.

#### Refractive Index
- **`fast_n`**: Quick estimate of the refractive index.
- **`n_extracted`**: Extracted refractive index values (complex).

#### Material Properties
- **`Length`**: Thickness of the material (in meters).
- **`unwrapping_regression_range`**: Frequency range for phase unwrapping.

---

### Methods

#### Time Domain
- **`plot_time_domain()`**: Plots the reference and sample signals in the time domain.
- **`get_processed_data()`**: Returns a dataframe containing frequency, time, and signal data for the reference and sample.

#### Windowing
- **`window_tukey_trivial(window_width_ref, window_width_sample, parameter)`**: Applies Tukey windowing to time-domain signals with configurable parameters.

#### Frequency Domain
- **`fft_signals(interpolation)`**: Performs FFT on reference and sample signals, calculates the transfer function, and estimates a quick refractive index.
- **`get_fft_data()`**: Returns a dataframe containing frequency-domain data, including amplitude and phase of signals and transfer functions.
- **`plot_frequency_domain()`**: Plots amplitude and phase data in the frequency domain.

#### Refractive Index Calculation
- **`calculate_refractive_index(n_0, frequency_stop)`**: Uses the Newton-Raphson method to compute the refractive index up to a specified frequency limit.
- **`get_refractive_index_data()`**: Returns a dataframe with real and imaginary parts of the refractive index.
- **`plot_refractive_index()`**: Plots the real and imaginary parts of the refractive index against frequency.

---

## Notes
- Input data should conform to the format in the `Data_sets` folder.
- The `thickness` parameter must be provided in meters for correct calculations.

---

## Dependencies
- `numpy`
- `pandas`
- `matplotlib`