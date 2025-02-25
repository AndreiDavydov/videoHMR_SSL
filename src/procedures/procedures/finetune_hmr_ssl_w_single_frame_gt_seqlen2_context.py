from time import time
import os
import matplotlib.pyplot as plt

import torch

import src 
from src.datasets.datasets_common import UNNORMALIZE
from src.procedures.procedures_common import status_msg

from src.functional.renderer import (
    convert_vertices_to_mesh,
    fit_vertices_to_orthographic,
    get_vertex_visibility_mask,
    unproject_to_vertices,
)
from src.functional.hmr import hmr_inference
from src.models.hmr_temporal import hmr_inference_w_context
from pytorch3d.transforms import matrix_to_axis_angle

from src.utils.vis_utils import render_mesh_onto_image_batch, make_square_grid

from src.functional.optical_flow import unproject_optical_flows_to_vertices, get_of
from src.procedures.procedures.finetune_hmr_ssl_w_single_frame_gt import setup, train_frame
from src.procedures.procedures.eval_hmr import valid as eval_on_3dpw


def train_video(sample, trainer):
    ### mostly copied from `finetune_hmr_ssl.py` procedure
    device = trainer.device0

    ### turn off BN 
    trainer.models.hmrnet.eval()

    # take one video sequence: B x seqlen==ctx_len+1 x 3 x 224 x 224
    img = sample["video"]
    batch_size = img.size(0)
    img_size = img.size(-1)

    ### compute optical flows between last and second last frames
    of_forward = get_of(trainer.optical_flow_model, UNNORMALIZE(img[:,-2]), UNNORMALIZE(img[:,-1]), device)
    of_backward = get_of(trainer.optical_flow_model, UNNORMALIZE(img[:,-1]), UNNORMALIZE(img[:,-2]), device)

    ctx_len = trainer.models.filmnet.module.ctx_len
    img_wo_context = img[:, :ctx_len]

    ### prepare the context
    out = hmr_inference(
        img_wo_context.flatten(start_dim=0, end_dim=1).to(device,non_blocking=True), 
        trainer.models.hmrnet, 
        trainer.smpl_model_49
        )
    ctx_aa = matrix_to_axis_angle(out['smpl_rotmat']).flatten(start_dim=1)
    ctx = torch.cat((ctx_aa, out['smpl_shape'], out['camera']), dim=-1)
    ctx = ctx.view(batch_size, ctx_len, -1)
    verts3d = out['verts3d']
    verts3d_2nd_last = verts3d.view(batch_size, -1, verts3d.size(-2), 3)[:,-1] # take 2nd last verts

    gammas, betas = trainer.models.filmnet(ctx.detach())  # detach the context (?)

    ### inference with context
    out = hmr_inference_w_context(
        img[:, -1].to(device,non_blocking=True), 
        gammas, betas, 
        trainer.models.hmrnet.module, 
        trainer.smpl_model_49)

    verts3d_last = out['verts3d']

    unproj_flow2d_forward, vis_mask_forward = unproject_optical_flows_to_vertices(
        verts3d_2nd_last, of_forward, 
        trainer.smpl_model_faces, 
        trainer.cameras)
    vis_mask = vis_mask_forward  # B x N
    
    verts_flow2d_forward  = verts3d_last[..., :2] - verts3d_last[..., :2]

    loss = 0

    ### Flow 2d - compute only forward
    if trainer.losses_weights.flow_2d > 0:
        flow_2d_forward = trainer.losses.flow_2d(
            verts_flow2d_forward, unproj_flow2d_forward, vis_mask
        )
        flow_2d = flow_2d_forward
        loss += flow_2d * trainer.losses_weights.flow_2d
        trainer.meters.train.flow_2d.update_raw(flow_2d.item())

    return loss


def train(trainer):

    absolute_start = time()
    
    dl_len = len(trainer.dataload.multidl)
    for batch_idx, (sample_frame, sample_video) in enumerate(trainer.dataload.multidl, start=1):
        
        ### inference for training with GT per frame
        train_frame_loss = train_frame(sample_frame, trainer)

        ### inference for training with OFC guidance 
        train_video_loss = train_video(sample_video, trainer)

        ### weight frame / video terms
        loss = trainer.cfg.W_FRAME * train_frame_loss + trainer.cfg.W_OFC * train_video_loss

        ### do backprop
        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()

        trainer.meters.train.full.update(loss.item(), n=1)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.full, total_time)
        if batch_idx != dl_len: trainer.meters.train.full.epochends()
    

def valid(trainer):
    trainer.logger.info("=> Plot images of train_video dataset...")
    ### save intermediate figs
    sq_side = 5 
    
    dset = trainer.datasets.train_video
    dataload = torch.utils.data.DataLoader(dset, batch_size=sq_side**2, num_workers=0, shuffle=True)
    for sample in dataload:
        break
    img = sample['video'][:,0]

    with torch.no_grad():
        out = hmr_inference(
            img.to(trainer.device0), 
            trainer.models.hmrnet, 
            trainer.smpl_model_49
        )
    
    rendered_imgs = render_mesh_onto_image_batch(
        UNNORMALIZE(img), 
        out["verts3d"].detach(), 
        faces=trainer.smpl_model_faces.repeat(img.size(0),1,1).to(trainer.device0), 
        device=trainer.device0)

    rendered_imgs = make_square_grid(rendered_imgs, sq_side=sq_side)

    ### save in folder figs/epoch_<epoch>.png
    savepath = os.path.join(trainer.final_output_dir, "figs")
    os.makedirs(savepath, exist_ok=True)

    savepath = os.path.join(savepath, f"ep_{trainer.cur_epoch:05}.png")
    plt.imsave(savepath, rendered_imgs.numpy())
    trainer.logger.info(f"=> saved in {savepath}")


    ### evaluation on 3DPW
    trainer.logger.info("=> Evaluate on 3DPW test set...")
    pa_mpjpe = eval_on_3dpw(trainer)

    return pa_mpjpe
