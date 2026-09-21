# HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering

[Project page](https://timsong412.github.io/HDGS-ProjPage/) 

The codebase is build upon [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)

![Teaser Video](assets/teaser.gif)

Demo code for **HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering**

The `main` branch contains k-sort only module, without textured surfels. The `texture` banch contains the textured surfel design, whose training relies on the intermediate checkpoints of the non-texture branch.

This is the `texture` branch

## Installation

Set the macro `SORT_WINDOW` in `submodules/diff-surfel-rasterization/cuda_rasterizer/auxiliary.h` to determine the sorting buffer size before compiling the rasterizer.

```bash
# download
git clone https://github.com/TimSong412/HDGS.git --recursive
git checkout texture
# if you have an environment used for 3dgs, use it
# if not, create a new environment
conda env create --file environment.yml
conda activate hdgs
# compile and reinstall the rasterizer if modified
bash update_pkg.sh
```
Prepare the data as 3D Gaussian Splatting under `dataset` dir.

## Training
You should train textured splatting upon the pre-trained and pruned checkpoints. The example training command is in `run_360.py` for the nerf360 dataset.


## Citation
If you find our code or paper helps, please consider citing:
```bibtex
@article{song2024hdgs,
  title={Hdgs: Textured 2d gaussian splatting for enhanced scene rendering},
  author={Song, Yunzhou and Lin, Heguang and Lei, Jiahui and Liu, Lingjie and Daniilidis, Kostas},
  journal={arXiv preprint arXiv:2412.01823},
  year={2024}
}
```
