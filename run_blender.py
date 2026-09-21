import os
from pathlib import Path

all_scenes = sorted(Path("dataset/nerf_synthetic").glob("*"))

for scene in all_scenes:
    if not scene.is_dir():
        continue
    scene_name = scene.stem
    cmd = f"python train.py -s {scene.__str__()} -m evalN01/{scene_name} --depth_ratio 1.0 --lambda_dist 100 --eval --densify_grad_threshold 0.0002 --densification_interval 150 --densify_until_iter 20000 --opacity_cull 0.01 --lambda_normal 0.01 --white_background --iteration 40000"
    print(cmd)
    os.system(cmd)