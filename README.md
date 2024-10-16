# THz-TDS

This project aims to provide a toolkit for material parameter extraction using THz-TDS.
The current implementation uses a basic Newton-Raphson method on an experimental transfer function for a no refections theoretical transfer function.

## Features
### Extraction
- Preprocessing of time-domain data
- FFT-based signal extraction
- Refractive index calculations
- Plotting functionalities

## Folders and files
### Extraction (folder)
Contains:
- Extractor.py: hold the class 'Extractor', currently this is a mono class containing all methods for transforming time domian data and extracting material parameters.
- transfer_functions.py: small file containing definitions for the transfer functions. currently only contains definitions for 0 reflection model.


## Requirements
- Python 3.9
- Numpy
- Pandas
- Scikit-learn
- Matplotlib

## Data requirements
Data being input to the Extractor class/ methods should follow the schematic and structure of the example data sets in the 'Data_sets' folder


## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/THz-TDS.git