# Thermal and RGB Image Alignment Tool

This tool aligns thermal images with their corresponding RGB images while preserving aspect ratios and image quality. It's particularly useful for applications requiring multi-spectral image analysis, such as thermal imaging, surveillance, and computer vision tasks.

## Features

- **Aspect Ratio Preservation**: Maintains the original aspect ratio of thermal images during alignment
- **Feature-Based Alignment**: Uses ORB (Oriented FAST and Rotated BRIEF) feature detection for accurate image registration
- **Automatic Processing**: Processes entire directories of image pairs with a single command
- **Flexible Output**: Saves both aligned thermal and RGB images for further analysis
- **Configurable Parameters**: Adjustable parameters for feature matching and image processing

## Prerequisites

- Python 3.9+
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Task1
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install them manually:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

### Command Line Interface

```bash
python thermal/task_1_code.py --input <input_directory> --output <output_directory>
```

### Arguments

- `--input`: Path to the input directory containing thermal and RGB image pairs
- `--output`: Path to the directory where processed images will be saved
- `--min-matches`: (Optional) Minimum number of feature matches required for alignment (default: 4)

### Input Format

Place your image pairs in the input directory with the following naming convention:
- RGB images: `*_rgb.jpg` or `*_RGB.jpg`
- Thermal images: `*_thermal.jpg` or `*_THERMAL.jpg`

## How It Works

### 1. Image Loading and Preprocessing
- Both thermal and RGB images are loaded in color
- RGB image is converted to grayscale for feature detection
- Thermal image is processed while preserving its color information

### 2. Scaling and Aspect Ratio Handling
- The thermal image is proportionally scaled to fit within the RGB image dimensions
- A small margin (3% of image dimensions) is maintained around the thermal image
- Minimum padding of 10 pixels is ensured on all sides

### 3. Feature Detection and Matching
- ORB (Oriented FAST and Rotated BRIEF) features are detected in both images
- Features are matched using Brute-Force matcher with Hamming distance
- Matches are filtered using the Lowe's ratio test to remove outliers

### 4. Homography Estimation
- A homography matrix is computed using RANSAC (Random Sample Consensus)
- The thermal image is warped to align with the RGB image using the computed homography

### 5. Output Generation
- Aligned thermal and RGB images are saved to the output directory
- Both original and processed images are preserved for comparison

## Output

The processed images are saved in the specified output directory with the following structure:

```
output_directory/
├── aligned_thermal_<original_name>.jpg
├── original_rgb_<original_name>.jpg
└── aligned_rgb_<original_name>.jpg
```

## Example

```bash
python thermal/task_1_code.py --input ./input_images --output ./aligned_results --min-matches 10
```

## Troubleshooting

- **No matches found**: Try adjusting the `--min-matches` parameter or ensure good feature points in both images
- **Poor alignment**: Check if the image pairs are properly named and correspond to the same scene
- **Memory issues**: Reduce image resolution if processing large datasets


