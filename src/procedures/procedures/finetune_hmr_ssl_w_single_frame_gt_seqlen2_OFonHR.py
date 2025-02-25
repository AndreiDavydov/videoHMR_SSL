from time import time
import os
import matplotlib.pyplot as plt

import torch

import src 
from src.datasets.datasets_common import UNNORMALIZE
from src.procedures.procedures_common import status_msg

from src.functional.hmr import hmr_inference
from src.functional.optical_flow import unproject_optical_flows_to_vertices, get_of
from src.procedures.procedures.finetune_hmr_ssl_w_single_frame_gt import setup, train_frame
from src.procedures.procedures.finetune_hmr_ssl_w_single_frame_gt_seqlen2 import valid


def train_video(sample, trainer):
    ### mostly copied from `finetune_hmr_ssl.py` procedure
    device = trainer.device0

    ### turn off BN 
    trainer.models.hmrnet.eval()

    # take one video sequence: B x seqlen==2 x 3 x H x W
    img = sample["videoOF"]
    batch_size = img.size(0)
    img_HR_size = img.size(-1)

    ### compute optical flows
    of_forward = get_of(trainer.optical_flow_model, UNNORMALIZE(img[:,0]), UNNORMALIZE(img[:,1]), device)
    of_backward = get_of(trainer.optical_flow_model, UNNORMALIZE(img[:,1]), UNNORMALIZE(img[:,0]), device)

    ### inference
    img = sample["video"]
    img_LR_size = img.size(-1)
    out = hmr_inference(
        img.flatten(start_dim=0, end_dim=1).to(device,non_blocking=True), 
        trainer.models.hmrnet, 
        trainer.smpl_model_49
        )
    verts3d = out['verts3d']
    verts3d = verts3d.view(batch_size, 2, verts3d.size(-2), 3) # aligned with low-res images

    s = img_HR_size / img_LR_size

    unproj_flow2d_forward, vis_mask_forward   = unproject_optical_flows_to_vertices(verts3d[:,0] * s, of_forward, trainer.smpl_model_faces, trainer.cameras)
    unproj_flow2d_backward, vis_mask_backward = unproject_optical_flows_to_vertices(verts3d[:,1] * s, of_backward, trainer.smpl_model_faces, trainer.cameras)
    unproj_flow2d_forward = unproj_flow2d_forward / s
    unproj_flow2d_backward = unproj_flow2d_backward / s
    ###  mask should be the intersection
    vis_mask = vis_mask_forward * vis_mask_backward  # B x N
    
    verts_flow2d_forward  = verts3d[:, -1, :, :2] - verts3d[:, 0, :, :2]
    verts_flow2d_backward = verts3d[:, 0, :, :2] - verts3d[:, -1, :, :2]

    loss = 0

    ### Flow 2d - compute in both time-directions and average
    if trainer.losses_weights.flow_2d > 0:
        flow_2d_forward = trainer.losses.flow_2d(
            verts_flow2d_forward, unproj_flow2d_forward, vis_mask
        )
        flow_2d_backward = trainer.losses.flow_2d(
            verts_flow2d_backward, unproj_flow2d_backward, vis_mask
        )
        flow_2d = (flow_2d_forward + flow_2d_backward) / 2
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