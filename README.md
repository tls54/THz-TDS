# THz-TDS

This project aims to provide a toolkit for material parameter extraction using THz-TDS.
The current implementation uses a basic Newton-Raphson method on an experimental transfer function for a no refections theoretical transfer function.

## Features
### Extraction
- Preprocessing of time-domain data
- FFT-based signal extraction
- Refractive index calculations
- Plotting functionalities

## Package structure
### Extraction (Package)
Contains:
- __init__.py: Empty init file that allows the Extraction folder to be treated as a package.
- Extractor.py: Hold the class 'Extractor'. This hold the main definition of the class, provides a single point that can be called to perform extraction.
- transfer_functions.py: Small file containing definitions for the transfer functions. currently only contains definitions for 0 reflection model.
- plotting.py: Holds the code for the plotting procedure, provides a consistent format for presenting data.
- constants.py: Houses consistent values for physical constants across the package.
- transformations.py: Houses scripts for complex data transformations such as FFTs. We will attempt to add the Newton-Raphson method to this file.



## Requirements
- Python 3.9+
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- ipympl - (for interactive plots in notebooks, This is recomended)

## Data requirements
Data being input to the Extractor class/ methods should follow the schematic and structure of the example data sets in the 'Data_sets' folder.


## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/THz-TDS.git