from time import time

import torch

import src 
from src.datasets.datasets_common import UNNORMALIZE

from src.functional import smpl
from src.functional.optical_flow import unproject_optical_flows_to_vertices
from src.functional.renderer import get_default_cameras
from src.models import raft

from src.procedures.procedures.eval_hmr import valid as eval_on_3dpw
from src.procedures.procedures_common import status_msg

import matplotlib.pyplot as plt
from src.datasets.datasets_common import UNNORMALIZE
from src.utils.vis_utils import render_mesh_onto_image
import numpy as np

import os


def setup(trainer):
    device = trainer.device0

    ### init smpl
    trainer.smpl_model_14 = smpl.get_smpl_model("h36m", device=device)
    trainer.smpl_model_49 = smpl.get_smpl_model("extra", device=device)
    trainer.smpl_model_faces = trainer.smpl_model_49.faces.astype(int)

    # this renaming allows to run eval on 3dpw without any changes
    trainer.smpl_model = trainer.smpl_model_14

    ### set faces for renderer
    smpl_model_faces = trainer.smpl_model_49.faces.astype(int)
    trainer.smpl_model_faces = torch.tensor(smpl_model_faces.copy()).unsqueeze(0)

    ### init cameras
    trainer.cameras = get_default_cameras(device, mode="orthographic")

    ### init optical flow model
    optical_flow_model = raft.get_raft_pretrained().to(device)
    optical_flow_model.eval()
    trainer.optical_flow_model = optical_flow_model

    ### init HMR-best model (COCO-All EFT model, pa-mpjpe: 58.2 mm)
    hmr_best = src.models.hmr.get_hmr(pretrained=True)
    ckpt = "/cvlabdata2/home/davydov/videoHMR_SSL/eft_model_zoo/coco-all.pt"
    ckpt = torch.load(ckpt, map_location="cpu")["hmrnet_state_dict"]
    hmr_best.load_state_dict(ckpt)
    hmr_best = hmr_best.to(device)
    hmr_best.eval()
    trainer.hmr_best = hmr_best


def compute_optical_flows(model, video, device):
    of_input = UNNORMALIZE(video).to(device)
    img1 = of_input[:-1]  # start frames
    img2 = of_input[1:]  # end frames

    with torch.no_grad():
        # compute forward optical flow
        _, opt_flow_forward = model(img1, img2, iters=20, test_mode=True)
        # compute backward optical flow
        _, opt_flow_backward = model(img2, img1, iters=20, test_mode=True)
    return opt_flow_forward, opt_flow_backward


def train(trainer):

    absolute_start = time()
    device = trainer.device0

    ### turn off BN 
    trainer.models.hmrnet.eval()
    
    dl_len = len(trainer.dataload.video_finetune)   
    for batch_idx, sample in enumerate(trainer.dataload.video_finetune, start=1):
        # take one video sequence: seqlen x 3 x 224 x 224
        img = sample["video"][0]  # (batch_size is 1)
        batch_size = img.size(0)
        img_size = img.size(-1)

        ### compute optical flows
        opt_flow_forward, opt_flow_backward = compute_optical_flows(
            trainer.optical_flow_model, img, device
        )

        img = img.to(device, non_blocking=True)

        ### inference with pretrained HMR-best model
        with torch.no_grad():
            pred_smpl_rotmat_best, pred_smpl_shape_best, pred_camera_best = trainer.hmr_best(img)
            pred_out_best = trainer.smpl_model_49(
                betas=pred_smpl_shape_best,
                body_pose=pred_smpl_rotmat_best[:, 1:],
                global_orient=pred_smpl_rotmat_best[:, :1],
                pose2rot=False,
            )

            j3d_best = pred_out_best.joints
            verts3d_best = pred_out_best.vertices

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
        # scale, trans = pred_camera[:, 0], pred_camera[:, 1:]
        # use gt camera params!
        scale, trans = pred_camera_best[:, 0].detach(), pred_camera_best[:, 1:].detach() 

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
            trainer.meters.train.flow_2d.update(flow_2d.item(), n=1)
            trainer.meters.train.flow_2d.epochends()

        ### temporal smoothing part
        ### shape_smoothing
        if trainer.losses_weights.shape_smooth > 0:
            shape_smooth = trainer.losses.shape_smooth(pred_smpl_shape, pred_smpl_shape_best)
            loss += shape_smooth * trainer.losses_weights.shape_smooth
            trainer.meters.train.shape_smooth.update(shape_smooth.item(), n=1)
            trainer.meters.train.shape_smooth.epochends()

        ### pose 3d smoothing
        if trainer.losses_weights.j3d_smooth > 0:
            j3d_smooth = trainer.losses.j3d_smooth(j3d, j3d_best)
            loss += j3d_smooth * trainer.losses_weights.j3d_smooth
            trainer.meters.train.j3d_smooth.update(j3d_smooth.item(), n=1)
            trainer.meters.train.j3d_smooth.epochends()

        ### consistency
        if trainer.losses_weights.j3d_cons > 0:
            j3d_cons = ((j3d - j3d_best) ** 2).mean()
            loss += j3d_cons * trainer.losses_weights.j3d_cons
            trainer.meters.train.j3d_cons.update(j3d_cons.item(), n=batch_size)
            trainer.meters.train.j3d_cons.epochends()

        if trainer.losses_weights.shape_cons > 0:
            shape_cons = ((pred_smpl_shape - pred_smpl_shape_best) ** 2).mean()
            loss += shape_cons * trainer.losses_weights.shape_cons
            trainer.meters.train.shape_cons.update(shape_cons.item(), n=batch_size)
            trainer.meters.train.shape_cons.epochends()

        ### do backprop
        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()

        trainer.meters.train.full.update(loss.item(), n=batch_size)

        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.full, total_time)
        trainer.meters.train.full.epochends()


def valid(trainer):
    trainer.logger.info("=> save intermediate seq figs...")
    ### save intermediate figs
    seq_ids = [10, 100, 500, 1001]

    for seq_idx in seq_ids:
        trainer.logger.info(f"\t=> seq idx {seq_idx:07}...")
        ### load sequence
        img = trainer.datasets.eval[seq_idx]["video"].to(trainer.device0)
        img_size = img.size(-1)

        ### inference
        with torch.no_grad():
            pred_smpl_rotmat, pred_smpl_shape, _ = trainer.models.hmrnet(img)
            _, _, pred_camera_best = trainer.hmr_best(img)
        
        pred_out = trainer.smpl_model_49(
            betas=pred_smpl_shape,
            body_pose=pred_smpl_rotmat[:, 1:],
            global_orient=pred_smpl_rotmat[:, :1],
            pose2rot=False,
        )
        j3d = pred_out.joints
        verts3d = pred_out.vertices

        ### align vertices with pixels
        # use gt camera params!
        scale, trans = pred_camera_best[:, 0].detach(), pred_camera_best[:, 1:].detach() 
        verts3d = verts3d * scale.view(verts3d.size(0), 1, 1)
        verts3d[:, :, 0:2] = verts3d[:, :, 0:2] + trans.view(verts3d.size(0), 1, 2)
        verts3d = (verts3d + 1) / 2 * img_size

        ### render sequence
        full_fig = []
        take_each = 5
        for i, (img_, verts3d_) in enumerate(
            zip(
                UNNORMALIZE(img.cpu()[::take_each]), 
                verts3d.cpu()[::take_each]
                )
            ):
            img_ = img_.permute(1,2,0)
            blend_img_ = render_mesh_onto_image(
                img_.numpy(), verts3d_.detach().numpy(), trainer.smpl_model_faces[0])
            full_fig.append(blend_img_)

        full_fig = np.concatenate(full_fig, axis=1)

        ### save in folder figs/seq_<seq_idx>/epoch_<epoch>.png
        savepath = os.path.join(trainer.final_output_dir, "figs", f"seq_{seq_idx:07}")
        os.makedirs(savepath, exist_ok=True)

        savepath = os.path.join(savepath, f"epoch{trainer.cur_epoch:05}.png")
        plt.imsave(savepath, full_fig)


    ### evaluation on 3DPW
    trainer.logger.info("=> Evaluate on 3DPW test set...")
    pa_mpjpe = eval_on_3dpw(trainer)

    return pa_mpjpe
