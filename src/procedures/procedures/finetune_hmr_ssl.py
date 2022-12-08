from time import time

import torch

from src.datasets.datasets_common import UNNORMALIZE

from src.functional import smpl
from src.functional.optical_flow import unproject_optical_flows_to_vertices
from src.functional.renderer import get_default_cameras
from src.models import raft

from src.procedures.procedures.eval_hmr import valid as eval_on_3dpw
from src.procedures.procedures_common import status_msg


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
    optical_flow_model = raft.raft_pretrained().to(device)
    optical_flow_model.eval()
    trainer.models.optical_flow_model = optical_flow_model


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


import sys


def train(trainer):

    absolute_start = time()
    device = trainer.device0

    dl_len = len(trainer.dataload.video_finetune)
    for batch_idx, sample in enumerate(trainer.dataload.video_finetune, start=1):
        # take one video sequence 32 x 3 x 224 x 224
        img = sample["video"][0]  # (batch_size is 1)
        batch_size = img.size(0)
        img_size = img.size(-1)

        ### compute optical flows
        opt_flow_forward, opt_flow_backward = compute_optical_flows(
            trainer.models.optical_flow_model, img, device
        )

        ### inference
        img = img.to(device, non_blocking=True)
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
        scale, trans = pred_camera[:, 0], pred_camera[:, 1:]
        verts3d = verts3d * scale.view(verts3d.size(0), 1, 1)
        verts3d[:, :, 0:2] = verts3d[:, :, 0:2] + trans.view(verts3d.size(0), 1, 2)
        verts3d = (verts3d + 1) / 2 * img_size

        if torch.isnan(verts3d).sum() > 0:
            print("verts has NaNs! before optical flow")
            print(verts3d.sum())
            print(img.sum())
            print(pred_smpl_rotmat.sum())
            print(pred_smpl_shape.sum())
            print(pred_camera.sum())
            sys.exit()

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

        if (
            torch.isnan(unproj_flow2d_forward).sum() > 0
            or torch.isnan(unproj_flow2d_backward).sum() > 0
        ):
            print("unproj flows have NaNs!")
            print(unproj_flow2d_forward.sum())
            print(unproj_flow2d_backward.sum())
            sys.exit()

        if vis_mask.sum() == 0:
            print("vis_mask is 0 everywhere!!!")
            torch.save(
                dict(
                    verts3d=verts3d,
                    opt_flow_backward=opt_flow_backward,
                    opt_flow_forward=opt_flow_forward,
                ),
                "/tmp/DUMP.pth",
            )
            
            print(
                "saved to/tmp/DUMP.pth"
            )
            sys.exit()

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

        if torch.isnan(loss).sum() > 0:
            print("flow2d loss is NaN, has no idea why!")
            sys.exit()

        ### temporal smoothing part
        ### shape_smoothing
        if trainer.losses_weights.shape_smooth > 0:
            shape_smooth = trainer.losses.shape_smooth(pred_smpl_shape, pred_smpl_shape)
            loss += shape_smooth * trainer.losses_weights.shape_smooth
            trainer.meters.train.shape_smooth.update(shape_smooth.item(), n=1)

        if torch.isnan(loss).sum() > 0:
            print("shape_smooth loss is NaN, has no idea why!")
            sys.exit()

        ### pose 3d smoothing
        if trainer.losses_weights.j3d_smooth > 0:
            j3d_smooth = trainer.losses.j3d_smooth(j3d, j3d)
            loss += j3d_smooth * trainer.losses_weights.j3d_smooth
            trainer.meters.train.j3d_smooth.update(j3d_smooth.item(), n=1)

        if torch.isnan(loss).sum() > 0:
            print("shape_smooth loss is NaN, has no idea why!")
            sys.exit()

        ### do backprop
        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()

        trainer.meters.train.full.update(loss.item(), n=batch_size)

        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.full, total_time)


def valid(trainer):
    ### evaluation on 3DPW
    pa_mpjpe = eval_on_3dpw(trainer)

    return pa_mpjpe
