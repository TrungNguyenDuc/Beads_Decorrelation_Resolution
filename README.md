# Bead Resolution Evaluation for Oblique Plane Microscopy

This repository contains Python code for evaluating the resolution of fluorescent beads with and without an autofocusing system in an oblique plane microscopy setup. 
The code implements an image decorrelation algorithm to estimate signal-to-noise ratio (SNR) and resolution metrics for 2D images and image stacks. See our bioRxiv paper [1] (https://doi.org/10.1101/2024.11.29.626121) for full optical setup instructions.

## Features

- **Image Decorrelation Algorithm**: Implements the Descloux et al. image decorrelation method to estimate resolution and SNR.
- **Support for 2D and 3D Data**: Processes single images or stacks of images with specified pixel sizes.
- **Parallel Processing**: Utilizes multi-threading for efficient computation.
- **Customizable Parameters**: Allows users to adjust parameters such as pixel size, apodization settings, and Gaussian filter widths.
- **Visualization Path**: Provides visual outputs to compare bead resolution over time with and without autofocusing.

## Requirements

This project requires Python 3.7 or higher and the following Python libraries:

- `numpy`
- `tifffile`
- `scipy`
- `matplotlib`
- `Pillow`
- `joblib`
- `tkinter`

Install the dependencies using the following command:

```bash
pip install numpy tifffile scipy matplotlib Pillow joblib
```

## Usage

### 1. Preparing the Data

Ensure that your microscopy images are saved as TIFF files. The program supports both single 2D images and multi-plane image stacks.

### 2. Running the Script

1. Open the Python script and adjust the parameters as needed:
   - `lateral_pixel_size`: Lateral pixel size in nanometers (default: 147 nm).
   - `axial_pixel_size`: Axial pixel size in nanometers (default: 147 nm).
2. Execute the script to load your data and analyze resolution and SNR.

### 3. Output

The script generates:

- Resolution and SNR values for each image plane.
- Optional visualization of the decorrelation results.
- **Visualization Path**: Visual comparison of the bead sample with and without autofocusing, demonstrating that resolution is maintained over time when autofocusing is enabled.

## Classes and Methods

### **`ImageDecorr`**

Processes a 2D image and calculates its resolution and SNR using the decorrelation algorithm.

#### Key Methods:

- `__init__(image, pixel_size, square_crop)`: Initializes the class with image data and parameters.
- `compute_resolution()`: Computes the optimal resolution using the geometric mean of SNR and cutoff radius.
- `all_corcoefs(num_rs, r_min, r_max, num_ws)`: Computes correlation coefficients for different radii and Gaussian filter widths.

### **`StackImDecorr`**

Processes an n-dimensional image stack to calculate resolution and SNR for each plane.

#### Key Methods:

- `__init__(stack, pixel_size, axes, square_crop)`: Initializes the class with a multi-plane image stack.
- `measure(parallel, n_jobs)`: Performs resolution estimation across all planes, optionally in parallel.
- `to_csv(filename)`: Saves the results to a CSV file.

## Example

```python
import numpy as np
from tifffile import imread
from your_script import StackImDecorr

# Load a multi-plane image stack
image_stack = imread('path_to_tiff_stack.tif')

# Initialize the resolution estimator
decorrelation = StackImDecorr(image_stack, pixel_size=147, axes="ZXY")

# Perform resolution estimation
decorrelation.measure(parallel=True, n_jobs=8)

# Save results to CSV
decorrelation.to_csv('results.csv')
```
![OPM-Autofocus-Fig S4](https://github.com/user-attachments/assets/08b64cb8-6fd3-44b9-9519-cca645281cb7)

## References

The code was used to evaluate the Autofocusing system for OPM microscopy as shown in the paper:

- Nguyen, T. D., Rahmani, A., Ponjavic, A., Millett-Sikking, A., & Fiolka, R. (2024). Active Remote Focus Stabilization in Oblique Plane Microscopy. bioRxiv (https://doi.org/10.1101/2024.11.29.626121).

The code implements methods based on the paper:

- Descloux, Adrien, et al. Combined multi-plane phase retrieval and super-resolution optical fluctuation imaging for 4D cell microscopy. Nature Photonics (2018).



