# THz-TDS

This project aims to provide a toolkit for material parameter extraction using THz-TDS.
The current implementation uses a basic Newton-Raphson method on an experimental transfer function for a no reflections theoretical transfer function.



## Features
### Extraction
- Preprocessing of time-domain data
- FFT-based signal extraction
- Refractive index calculations
- Plotting functionalities

### Data_sets
- `simple_data`: Contains a reference and clean time domain data for silicon at 300 micron thickness
- `complex data`: Contains a reference pulse (ref2.csv) and 4 samples, two GaAs and two LiNbO. The thickness is the second number in the file name and is given in microns.

### Notebooks
- `testing.ipynb`: Basic demonstration of class implementation for a simple dataset.
- `Extractions` (folder): Contains notebooks with material parameter extractions for all samples. 

### Reversed_algo
- Functions and testing related to reversing the transfer function algorithm.
- This allows us to generate a transfer unction for a given refractive index or refractive index distribution.
- Using this we can generate datasets to train a neural network to learn to extract material parameters from a transfer function.


## Package structure
### Extraction (Package)
#### Summary of Contents
- **`__init__.py`**: Empty init file that allows the Extraction folder to be treated as a package.
- **`Extractor.py`**: Hold the class 'Extractor'. This hold the main definition of the class, provides a single point that can be called to perform extraction.
- **`transfer_functions.py`**: Small file containing definitions for the transfer functions. currently only contains definitions for 0 reflection model.
- **`plotting.py`**: Holds the code for the plotting procedure, provides a consistent format for presenting data.
- **`constants.py`**: Stores consistent values for physical constants across the package.
- **`transformations.py`**: Houses scripts for complex data transformations such as FFTs. We will attempt to add the Newton-Raphson method to this file.



## Requirements
- **`Python 3.9+`**
- **`Numpy`**
- **`Pandas`**
- **`Scikit-learn`**
- **`Matplotlib`**
- **`ipympl`** - This is for interactive plots in notebooks, this is recommended.

## Data requirements
Data being input to the Extractor class/ methods should follow the schematic and structure of the example data sets in the 'Data_sets' folder.



## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/THz-TDS.git