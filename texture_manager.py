import torch
import numpy as np
import torch.nn as nn
from scene.gaussian_model import GaussianModel
import tqdm
from utils.loss_utils import l1_loss, ssim
import threading



class TextureBuffer(nn.Module):
    def __init__(self, buffer_size, init_color=None, devices='cuda'):
        '''
        init_color: tensor of shape (buffer_size, 3, 3)
        '''
        super().__init__()
        self.buffer_size = buffer_size
        self.devices = devices
        print("buffer_size: ", buffer_size)
        if init_color is None:
            self.texture_buffer = nn.parameter.Parameter(torch.zeros(buffer_size, 3), requires_grad=True)
        else:
            self.texture_buffer = nn.parameter.Parameter(init_color, requires_grad=True)

def add_checkerboard(texture_buffer, texture_index, checkerboard_size=10):
    # add checkboard pattern
    for gid in tqdm.trange(len(texture_index), desc="add checkboard pattern"):
        start, W, H = texture_index[gid]
        checkerboard = torch.zeros(H, W, 3).to(texture_buffer.device)
        # each checkerboard is 10x10
        for i in range(0, H//10):
            for j in range(0, W//10):
                if (i+j) % 2 == 0:
                    checkerboard[i*10:(i+1)*10, j*10:(j+1)*10] = 1.0
        texture_buffer[start:start+W*H] = checkerboard.reshape(-1, 3)
    return texture_buffer
    

def init_texture(gaussans: GaussianModel, pix_scale=100, return_buffer_index=False, white_background=False, max_tex = 500):
    '''
    init a texture buffer with gaussians
    each gaussian gives a index, start, h, w

    '''
    gs_scales = gaussans.get_scaling * pix_scale
    gs_W = gs_scales[..., 0]
    gs_H = gs_scales[..., 1]
    gs_W = torch.ceil(gs_W).long()
    gs_H = torch.ceil(gs_H).long()
    max_size = torch.maximum(gs_W, gs_H)
    overflow = max_size > max_tex
    gs_W[overflow] = gs_W[overflow] * max_tex // max_size[overflow]
    gs_H[overflow] = gs_H[overflow] * max_tex // max_size[overflow]
    print("max W: ", torch.max(gs_W).item())
    print("max H: ", torch.max(gs_H).item())
    gs_area = gs_W * gs_H
    gs_start = torch.cumsum(gs_area, dim=0)
    buffer_size = gs_start[-1].item()

    print("gaussian number: ", len(gs_start))

    gs_start = torch.cat([torch.tensor([0]).to(gs_start.device), gs_start[:-1]])
    gs_color = gaussans._features_dc
    if white_background:
        print("white background")
        texture_buffer = torch.ones(buffer_size, 3).to(gs_color.device)
    else:
        print("black background")
        texture_buffer = torch.zeros(buffer_size, 3).to(gs_color.device)
    
    texture_index = torch.stack([gs_start, gs_W, gs_H], dim=1).int()

    for gid in tqdm.trange(len(gs_start), desc="init texture"):
        start, W, H = texture_index[gid]
        texture_buffer[start:start+W*H] = gs_color[gid, 0].reshape(-1, 3).detach()

    # texture_buffer = add_checkerboard(texture_buffer, texture_index)

    # texture_buffer[:, 2] = 1e10
    # texture_buffer[:, 1] = 1e10
    # for i in range(len(gs_start)):
    #     texture_buffer[gs_start[i]:gs_start[i]+gs_area[i]] = gs_color[i, 0]
    texture_buffer = TextureBuffer(buffer_size, texture_buffer)

    if return_buffer_index:

        buffer_index = torch.zeros(buffer_size).to(gs_color.device).long()
        if len(buffer_index) == len(gs_start):
            buffer_index = torch.arange(len(gs_start)).to(gs_color.device).long()
        else:
            for gsid in range(len(gs_start)):
                buffer_index[gs_start[gsid]:gs_start[gsid]+gs_area[gsid]] = gsid

        return texture_index, texture_buffer.to(gs_color.device), buffer_index
    else:
        return texture_index, texture_buffer.to(gs_color.device)


def analyse_grad(texture_buffer:TextureBuffer, texture_index, buffer_index):
    '''
    texture index: N x 3, (start, W, H)
    '''
    N, _ = texture_index.shape
    all_grad = torch.zeros(N).to(texture_buffer.texture_buffer.device)
    
    all_grad.scatter_add_(0, buffer_index, texture_buffer.texture_buffer.grad.abs().sum(dim=1))

    return all_grad


def process_chunk(texture_buffer, new_texture_buffer, texture_index, new_texture_index, new_buffer_index, expand_gs, start, end):
    local_texture_start = new_texture_index[start, 0]
    local_texture_end = new_texture_index[end-1, 0] + new_texture_index[end-1, 1] * new_texture_index[end-1, 2]
    local_texture_buffer = torch.ones(local_texture_end - local_texture_start, 3).to(texture_buffer.texture_buffer.device)
    local_buffer_index = torch.zeros(local_texture_end - local_texture_start).to(texture_buffer.texture_buffer.device).long()
    for gsid in tqdm.trange(start, end):
        if expand_gs[gsid]:
            # upsample the texture
            start, W, H = texture_index[gsid]
            old_texture = texture_buffer.texture_buffer[start:start+W*H].reshape(H, W, 3).detach()
            new_texture = torch.nn.functional.interpolate(old_texture.permute(2, 0, 1).unsqueeze(0), (new_texture_index[gsid, 2], new_texture_index[gsid, 1]), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            # new_texture_buffer[new_texture_index[gsid, 0]:new_texture_index[gsid, 0]+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = new_texture.reshape(-1, 3)
            local_texture_buffer[new_texture_index[gsid, 0]-local_texture_start:new_texture_index[gsid, 0]-local_texture_start+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = new_texture.reshape(-1, 3)
            
        else:
            start, W, H = texture_index[gsid]
            # new_texture_buffer[new_texture_index[gsid, 0]:new_texture_index[gsid, 0]+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = texture_buffer.texture_buffer[start:start+W*H]
            local_texture_buffer[new_texture_index[gsid, 0]-local_texture_start:new_texture_index[gsid, 0]-local_texture_start+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = texture_buffer.texture_buffer[start:start+W*H]

        # new_buffer_index[new_texture_index[gsid, 0]:new_texture_index[gsid, 0]+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = gsid
        local_buffer_index[new_texture_index[gsid, 0]-local_texture_start:new_texture_index[gsid, 0]-local_texture_start+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = gsid
    new_texture_buffer[local_texture_start:local_texture_end] = local_texture_buffer
    new_buffer_index[local_texture_start:local_texture_end] = local_buffer_index
    

    
def expand_texture(texture_buffer:TextureBuffer, buffer_index, gaussians, optimizer:torch.optim.Optimizer, training_cams, pipe, background, render, opt, scale=1):
    '''
    texture index: N x 3, (start, W, H)
    '''
    print("expand texture")
    texture_index = gaussians.texture_index
    N, _ = texture_index.shape
    
    expand_gs = torch.zeros(N).bool().to(texture_buffer.texture_buffer.device)

    for cam in tqdm.tqdm(training_cams, desc='analyse gradient'):
        render_pkg = render(cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        optimizer.zero_grad()
        gaussians.optimizer.zero_grad(set_to_none = True)
        loss.backward()
        all_grad = analyse_grad(texture_buffer, texture_index, buffer_index)
        expand_gs = expand_gs | (all_grad > torch.mean(all_grad))
    
    optimizer.zero_grad()
    gaussians.optimizer.zero_grad(set_to_none = True)
    
    print("expand gaussians: ", torch.sum(expand_gs).item(), "/ ", N)
    
    with torch.no_grad():
        new_texture_index =texture_index.clone()
        new_texture_index[expand_gs, 1] = torch.ceil(gaussians.get_scaling[expand_gs, 0] * scale).int()
        new_texture_index[expand_gs, 2] = torch.ceil(gaussians.get_scaling[expand_gs, 1] * scale).int()
        all_area = new_texture_index[:, 1] * new_texture_index[:, 2]
        cum_area = torch.cumsum(all_area, dim=0)
        new_texture_index[1:, 0] = cum_area[:-1]
        new_texture_buffer = torch.ones(cum_area[-1], 3).to(texture_buffer.texture_buffer.device)
        new_buffer_index = torch.zeros(cum_area[-1]).to(texture_buffer.texture_buffer.device).long()
        # for gsid in tqdm.trange(N, desc=f'expand texture, scale: {scale}'):
        #     if expand_gs[gsid]:
        #         # upsample the texture
        #         start, W, H = texture_index[gsid]
        #         old_texture = texture_buffer.texture_buffer[start:start+W*H].reshape(H, W, 3).detach()
        #         new_texture = torch.nn.functional.interpolate(old_texture.permute(2, 0, 1).unsqueeze(0), (new_texture_index[gsid, 2], new_texture_index[gsid, 1]), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        #         new_texture_buffer[new_texture_index[gsid, 0]:new_texture_index[gsid, 0]+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = new_texture.reshape(-1, 3)
                
        #     else:
        #         start, W, H = texture_index[gsid]
        #         new_texture_buffer[new_texture_index[gsid, 0]:new_texture_index[gsid, 0]+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = texture_buffer.texture_buffer[start:start+W*H]

        #     new_buffer_index[new_texture_index[gsid, 0]:new_texture_index[gsid, 0]+new_texture_index[gsid, 1]*new_texture_index[gsid, 2]] = gsid
        chunk_size = 1000000
        threads = []
        for i in range(0, N, chunk_size):
            threads.append(threading.Thread(target=process_chunk, args=(texture_buffer, new_texture_buffer, texture_index, new_texture_index, new_buffer_index, expand_gs, i, min(i+chunk_size, N))))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


        del texture_buffer.texture_buffer
        torch.cuda.empty_cache()

    texture_buffer.texture_buffer = nn.parameter.Parameter(new_texture_buffer, requires_grad=True)
    
    # update the optimizer parameters
    store_state = optimizer.state.get(optimizer.param_groups[0]['params'][0], None)
    if store_state is not None:
        optimizer.state.pop(optimizer.param_groups[0]['params'][0])

    optimizer.param_groups[0]['params'][0] = texture_buffer.texture_buffer.requires_grad_(True)
    
    gaussians.texture_buffer = texture_buffer.texture_buffer
    gaussians.texture_index = new_texture_index
    return gaussians, optimizer, new_buffer_index


            
                    

