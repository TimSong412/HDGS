from pathlib import Path
import os

all_dtu = Path("dataset/DTU")
all_dtu = [x for x in all_dtu.iterdir() if x.is_dir()]

for dtu in all_dtu[8:]:
    scan_name = dtu.name
    cmd = f"python train.py -s {dtu} -m eval_DTUNVS/{scan_name} --depth_ratio 1.0 -r 2 --lambda_dist 1000 --eval --lambda_normal 0.04 -i images_rgb --densify_grad_threshold 0.0002 --densification_interval 150 --densify_until_iter 20000 --opacity_cull 0.01 --iteration 40000"
    print(cmd)
    os.system(cmd)