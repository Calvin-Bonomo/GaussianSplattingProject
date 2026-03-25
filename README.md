# Gaussian Splatting Exploration

## Motivation
This project was made for my machine learning class. I have always been 
interested in computer graphics, so, naturally, I decided that gaussian splatting 
would be a really interesting project to pursue. The relatively low cost of 
rasterizing tens or hundreds of thousands of spats from a single buffer rather 
than pathtracing possibly millions of tris for complex scenes is a really 
interesting optimization, yielding new avenues for real-time visualization of 
high fidelity scenes.

## Goals
- [ ] Build the initial gaussian splatting algorithm with a built-in training viewer
- [ ] Apply SOTA optimizations
- [ ] Look into novel deep-learning optimization techniques

## Requirements
This project requires:
- CUDA 12.8
- Python 3.13
- Conda

## Build Instructions

```bash
# Use conda to setup the environment
conda env create -f environment.yml

# Run tests
pytest

# Train scene on a dataset with a viewer
python run.py -s [PATH_TO_INPUT_IMAGES_DIR] -o [PATH_TO_OUTPUT_DIR]

# Train scene on a dataset without viewer
python run.py -s [PATH_TO_INPUT_IMAGES_DIR] -o [PATH_TO_OUTPUT_DIR] --no-viewer

```

## References
[1] Kerbl, B. et al. (2023) ‘3d Gaussian splatting for real-time radiance field rendering’, ACM Transactions on Graphics, 42(4), pp. 1–14. doi:10.1145/3592433. 

