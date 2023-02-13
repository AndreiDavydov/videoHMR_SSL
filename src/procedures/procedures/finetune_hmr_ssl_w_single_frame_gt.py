from os.path import join as ospjoin
from time import time

# import src.utils.manifold_utils as manifold

import os
import torch

import src 
from src.datasets.datasets_common import UNNORMALIZE

from src.functional import smpl
from src.functional.optical_flow import unproject_optical_flows_to_vertices
from src.functional.renderer import get_default_cameras
from src.models import raft

from src.procedures.procedures_common import status_msg
from src.utils.img_utils import convert_norm_points_to_bbox, weakProjection
from src.utils.plot_utils import plot_batch_with_mesh

import matplotlib.pyplot as plt
from src.datasets.datasets_common import UNNORMALIZE
from src.utils.vis_utils import render_mesh_onto_image
import numpy as np

import os

from src.procedures.procedures.finetune_hmr_ssl import compute_optical_flows, valid as valid_finetune, setup as setup_finetune


def setup(trainer):
    setup_finetune(trainer)


def train_frame(sample, trainer):
    ### copy from `train_hmr.py` procedure
    device = trainer.device0

    img = sample["img"].to(device, non_blocking=True)  # Bx3x224x224
    gt_kpts2d_norm = sample["gt2d_norm"].to(device, non_blocking=True)  # Bx49x3
    gt_smpl_shape = sample["smpl_shape"].to(device, non_blocking=True)  # Bx10
    gt_smpl_pose = sample["smpl_pose"].to(device, non_blocking=True)  # Bx72
    batch_size = img.size(0)

    gt_out = trainer.smpl_model_49(
            betas=gt_smpl_shape,
            body_pose=gt_smpl_pose[:, 3:],
            global_orient=gt_smpl_pose[:, :3],
        )
    gt_kpts3d = gt_out.joints.detach()  # Bx49x3
    # gt_vertices = gt_out.vertices  # Bx6890x3

    ### inference
    pred_smpl_rotmat, pred_smpl_shape, pred_camera = trainer.models.hmrnet(img)
    pred_out = trainer.smpl_model_49(
        betas=pred_smpl_shape,
        body_pose=pred_smpl_rotmat[:, 1:],
        global_orient=pred_smpl_rotmat[:, :1],
        pose2rot=False,
    )

    pred_kpts3d = pred_out.joints
    # pred_vertices = pred_out.vertices

    scale, trans = pred_camera[:, 0], pred_camera[:, 1:]
    pred_kpts2d_norm = weakProjection(pred_kpts3d, scale, trans)

    ### compute losses
    smpl_pose_loss = trainer.losses.smpl_pose_loss(pred_smpl_rotmat, gt_smpl_pose)
    smpl_shape_loss = trainer.losses.smpl_shape_loss(pred_smpl_shape, gt_smpl_shape)
    smpl_shape_prior_loss = trainer.losses.smpl_shape_prior_loss(pred_smpl_shape)
    kpts2d_loss = trainer.losses.kpts2d_loss(
        pred_kpts2d_norm[:, 25:], gt_kpts2d_norm[:, 25:]
    )
    kpts3d_loss = trainer.losses.kpts3d_loss(pred_kpts3d[:, 25:], gt_kpts3d[:, 25:])
    camera_scale_reg_loss = trainer.losses.camera_scale_reg_loss(scale)
    # vertex_loss = ... # NOTE in eft code, weight for vertex loss is 0

    full_loss = (
        trainer.losses_weights.smpl_pose_loss * smpl_pose_loss
        + trainer.losses_weights.smpl_shape_loss * smpl_shape_loss
        + trainer.losses_weights.smpl_shape_prior_loss * smpl_shape_prior_loss
        + trainer.losses_weights.kpts2d_loss * kpts2d_loss
        + trainer.losses_weights.kpts3d_loss * kpts3d_loss
        + trainer.losses_weights.camera_scale_reg_loss * camera_scale_reg_loss
    )
    full_loss *= 60

    trainer.meters.train.smpl_pose_loss.update_raw(smpl_pose_loss.item())
    trainer.meters.train.smpl_shape_loss.update_raw(smpl_shape_loss.item())
    trainer.meters.train.smpl_shape_prior_loss.update_raw(smpl_shape_prior_loss.item())

    trainer.meters.train.kpts2d_loss.update_raw(kpts2d_loss.item())
    trainer.meters.train.kpts3d_loss.update_raw(kpts3d_loss.item())
    trainer.meters.train.camera_scale_reg_loss.update_raw(camera_scale_reg_loss.item())
    
    return full_loss


def train_video(sample, trainer):
    ### mostly copied from `finetune_hmr_ssl.py` procedure
    device = trainer.device0

    ### turn off BN 
    trainer.models.hmrnet.eval()

    # take one video sequence: seqlen x 3 x 224 x 224
    img = sample["video"][0]  # (batch_size is 1)
    batch_size = img.size(0)
    img_size = img.size(-1)

    ### compute optical flows
    opt_flow_forward, opt_flow_backward = compute_optical_flows(
        trainer.optical_flow_model, img, device
    )

    img = img.to(device, non_blocking=True)

    ##########
    # replace consistency with training with GT!
    ##########
    # ### inference with pretrained HMR-best model
    # with torch.no_grad():
    #     pred_smpl_rotmat_best, pred_smpl_shape_best, pred_camera_best = trainer.hmr_best(img)
    #     pred_out_best = trainer.smpl_model_49(
    #         betas=pred_smpl_shape_best,
    #         body_pose=pred_smpl_rotmat_best[:, 1:],
    #         global_orient=pred_smpl_rotmat_best[:, :1],
    #         pose2rot=False,
    #     )

    #     j3d_best = pred_out_best.joints
    #     verts3d_best = pred_out_best.vertices

    ### inference
    pred_smpl_rotmat, pred_smpl_shape, pred_camera = trainer.models.hmrnet(img)
    pred_out = trainer.smpl_model_49(
        betas=pred_smpl_shape,
        body_pose=pred_smpl_rotmat[:, 1:],
        global_orient=pred_smpl_rotmat[:, :1],
        pose2rot=False,
    )

    j3d = pred_out.joints
    verts3d = pred_out.vertices

    ### align vertices with pixels

    # use predicted camera parameters
    scale, trans = pred_camera[:, 0], pred_camera[:, 1:]
    # use gt camera params!
    # scale, trans = pred_camera_best[:, 0].detach(), pred_camera_best[:, 1:].detach() 

    verts3d = verts3d * scale.view(verts3d.size(0), 1, 1)
    verts3d[:, :, 0:2] = verts3d[:, :, 0:2] + trans.view(verts3d.size(0), 1, 2)
    verts3d = (verts3d + 1) / 2 * img_size

    (
        unproj_flow2d_forward,
        unproj_flow2d_backward,
        vis_mask,
    ) = unproject_optical_flows_to_vertices(
        verts3d,
        opt_flow_forward,
        opt_flow_backward,
        trainer.smpl_model_faces,
        trainer.cameras,
    )
    
    verts_flow2d_forward = verts3d[1:, :, :2] - verts3d[:-1, :, :2]
    verts_flow2d_backward = verts3d[:-1, :, :2] - verts3d[1:, :, :2]

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

    # ### temporal smoothing part
    # ### shape_smoothing
    # if trainer.losses_weights.shape_smooth > 0:
    #     shape_smooth = trainer.losses.shape_smooth(pred_smpl_shape, pred_smpl_shape_best)
    #     loss += shape_smooth * trainer.losses_weights.shape_smooth
    #     trainer.meters.train.shape_smooth.update(shape_smooth.item(), n=1)
    #     trainer.meters.train.shape_smooth.epochends()

    # ### pose 3d smoothing
    # if trainer.losses_weights.j3d_smooth > 0:
    #     j3d_smooth = trainer.losses.j3d_smooth(j3d, j3d_best)
    #     loss += j3d_smooth * trainer.losses_weights.j3d_smooth
    #     trainer.meters.train.j3d_smooth.update(j3d_smooth.item(), n=1)
    #     trainer.meters.train.j3d_smooth.epochends()

    # ### consistency
    # if trainer.losses_weights.j3d_cons > 0:
    #     j3d_cons = ((j3d - j3d_best) ** 2).mean()
    #     loss += j3d_cons * trainer.losses_weights.j3d_cons
    #     trainer.meters.train.j3d_cons.update(j3d_cons.item(), n=batch_size)
    #     trainer.meters.train.j3d_cons.epochends()

    # if trainer.losses_weights.shape_cons > 0:
    #     shape_cons = ((pred_smpl_shape - pred_smpl_shape_best) ** 2).mean()
    #     loss += shape_cons * trainer.losses_weights.shape_cons
    #     trainer.meters.train.shape_cons.update(shape_cons.item(), n=batch_size)
    #     trainer.meters.train.shape_cons.epochends()

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
    return valid_finetune(trainer)
    