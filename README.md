# Gaussian Splatting Exploration

## Motivation
This project was made with the intention of exploring gaussian splatting and 
experimenting with the usage of LLMs as part of my workflow. In this demo, I 
will make a striped down version of the gaussian splatting implementation done 
in the paper by Kerbl, et al.. Really, the only feature that is missing from 
my implementation is the use of spherical harmonics for view-dependent color.
Spherical harmonics are still a bit out of my league, and so is most of the 
math in the paper.

## Goals
- [ ] Build the initial gaussian splatting algorithm
- [ ] Create a real-time viewer so that the training process can be watched
- [ ] Do a write-up about the project and any revelations about the use of LLMs

## Requirements
In order to run the following demo, you must have the following:
- Python 3.13
- A NVIDIA gpu which supports CUDA 12.8
- Conda (not technically, but you'll need to do a lot of setup yourself)

## Usage
```
// Create the conda environment using the provided environment.yml file
conda env create -f environment.yml

// Activate the environment
conda activate gaussian_splatting

// Build the cuda modules
cmake -S ./rasterizer -B ./rasterizer/build
cmake --build ./rasterizer

// Learn a scene with the built-in viewer
python gs_demo.py -s "[PATH_TO_YOUR_COLMAP_DATA_HERE]" -o "[PATH_TO_YOUR_OUTPUT_DIR_HERE]" 

// Learn a scene without the built-in viewer
python gs_demo.py -s "[PATH_TO_YOUR_COLMAP_DATA_HERE]" -o "[PATH_TO_YOUR_OUTPUT_DIR_HERE]" --no-viewer
```

## References
[1] Kerbl, B. et al. (2023) ‘3d Gaussian splatting for real-time radiance field rendering’, ACM Transactions on Graphics, 42(4), pp. 1–14. doi:10.1145/3592433. 

