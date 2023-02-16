from time import time

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

from src.procedures.procedures.finetune_hmr_ssl_w_single_frame_gt import setup, valid, train_frame


def get_of(model, frames_start, frames_end, device):
    """Compute optical flow using off-the-shelf model"""
    img1 = UNNORMALIZE(frames_start).to(device)
    img2 = UNNORMALIZE(frames_end).to(device)

    with torch.no_grad():
        # compute optical flow
        _, opt_flow = model(img1, img2, iters=20, test_mode=True)
    return opt_flow


def hmr_inference(img, hmrnet, smpl_model, device, img_size=224):
    img = img.to(device, non_blocking=True)
    smpl_rotmat, smpl_shape, camera = hmrnet(img)
    out = smpl_model(
        betas=smpl_shape,
        body_pose=smpl_rotmat[:, 1:],
        global_orient=smpl_rotmat[:, :1],
        pose2rot=False,
    )
    j3d = out.joints
    verts3d = out.vertices

    ### align vertices with pixels
    scale, trans = camera[:, 0], camera[:, 1:]

    verts3d = verts3d * scale.view(verts3d.size(0), 1, 1)
    verts3d[:, :, 0:2] = verts3d[:, :, 0:2] + trans.view(verts3d.size(0), 1, 2)
    verts3d = (verts3d + 1) / 2 * img_size
    out = dict(j3d=j3d, verts3d=verts3d)
    return out


def unproject_optical_flows_to_vertices(verts3d, of, faces, cameras):
    """
    Args:
        verts3d (torch.tensor) B x N x 3 - mesh vertices, start/end together.
            X,Y coordinates are in pixels.
        of (torch.tensor) B x 2 x H x W - optical flow
            (e.g., from the pretrained optical flow predictor)
        faces (torch.LongTensor) 1 x Ntri x 3 - faces indices for mesh
            Must be copied to batch size.
    Output:
        unproj_flow2d B x N x 2 - 2d flow unprojected on (assigned to) vertices
        vis_mask B x N - {0., 1.} mask of vertex visibility
    """
    ### map coordinates to NDC format
    img_size = of.size()[-2:]
    batch_size = verts3d.size(0)
    n_verts = verts3d.size(1)
    faces_batch = faces.repeat(batch_size, 1, 1).to(verts3d.device)
    verts3d = fit_vertices_to_orthographic(verts3d, img_size=img_size)
    meshes = convert_vertices_to_mesh(verts3d, faces_batch)

    ### compute visibility mask
    vis_mask = get_vertex_visibility_mask(meshes, cameras, img_size)
    vis_mask = vis_mask.view(verts3d.size(0), verts3d.size(1))

    ### unproject optical flow to vertices
    unproj_flow2d = unproject_to_vertices(of, verts3d)
    return unproj_flow2d, vis_mask


def train_video(sample, trainer):
    ### mostly copied from `finetune_hmr_ssl.py` procedure
    device = trainer.device0

    ### turn off BN 
    trainer.models.hmrnet.eval()

    # take one video sequence: B x seqlen==2 x 3 x 224 x 224
    img = sample["video"]
    batch_size = img.size(0)
    img_size = img.size(-1)

    ### compute optical flows
    of_forward = get_of(trainer.optical_flow_model, img[:,0], img[:,1], device)
    of_backward = get_of(trainer.optical_flow_model, img[:,1], img[:,0], device)

    ### inference
    out = hmr_inference(img.flatten(start_dim=0, end_dim=1), trainer.models.hmrnet, trainer.smpl_model_49, device)
    verts3d = out['verts3d']
    verts3d = verts3d.view(batch_size, 2, verts3d.size(-2), 3)

    unproj_flow2d_forward, vis_mask_forward   = unproject_optical_flows_to_vertices(verts3d[:,0], of_forward, trainer.smpl_model_faces, trainer.cameras)
    unproj_flow2d_backward, vis_mask_backward = unproject_optical_flows_to_vertices(verts3d[:,1], of_backward, trainer.smpl_model_faces, trainer.cameras)
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
    