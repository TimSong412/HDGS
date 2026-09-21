from pathlib import Path
import os

# l40 gs: raw, gs2d: texture

outdoor = ["garden", "bicycle", "flowers", "stump", "treehill"]
indoor = ["room", "counter", "kitchen", "bonsai"]

for scene in outdoor:
    source = Path("dataset/360v2") / scene
    # cmd = f"python train.py -s {source} -i images_4 -m eval360S16/{scene} --eval --lambda_dist 100 --densify_grad_threshold 0.0002 --densification_interval 150 --densify_until_iter 20000 --opacity_cull 0.01 --lambda_normal 0.01 --iteration 40000"
    cmd = f"python train_texture.py -s {source} -i images_4 -m eval360S16/{scene}_tex200 --eval --lambda_dist 100 --densify_grad_threshold 0.0002 --densification_interval 150 --densify_until_iter 20000 --opacity_cull 0.01 --lambda_normal 0.01 --iteration 60000 --tex_res 200 --start_checkpoint eval360S16/{scene}/chkpnt40000.pth"
    print(cmd)
    os.system(cmd)

for scene in indoor:
    source = Path("dataset/360v2") / scene
    # cmd = f"python train.py -s {source} -i images_2 -m eval360S16/{scene} --eval --lambda_dist 100 --densify_grad_threshold 0.0002 --densification_interval 150 --densify_until_iter 20000 --opacity_cull 0.01 --lambda_normal 0.01 --iteration 40000"
    cmd = f"python train_texture.py -s {source} -i images_2 -m eval360S16/{scene}_tex500 --eval --lambda_dist 100 --densify_grad_threshold 0.0002 --densification_interval 150 --densify_until_iter 20000 --opacity_cull 0.01 --lambda_normal 0.01 --iteration 60000 --tex_res 500 --start_checkpoint eval360S16/{scene}/chkpnt40000.pth"
    print(cmd)
    os.system(cmd)