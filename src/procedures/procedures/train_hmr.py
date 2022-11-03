from os.path import join as ospjoin
from time import time

import src.utils.manifold_utils as manifold

import torch

from src.functional import smpl

from src.procedures.procedures.eval_hmr import valid as eval_on_3dpw

from src.procedures.procedures_common import status_msg
from src.utils.img_utils import convert_norm_points_to_bbox, weakProjection
from src.utils.plot_utils import plot_batch_with_mesh


def setup(trainer):
    ### init smpl
    trainer.smpl_model_14 = smpl.get_smpl_model("h36m", device=trainer.device0)
    trainer.smpl_model_49 = smpl.get_smpl_model("extra", device=trainer.device0)
    trainer.smpl_model_faces = trainer.smpl_model_49.faces.astype(int)

    # this renaming allows to run eval on 3dpw without any changes
    trainer.smpl_model = trainer.smpl_model_14


def train(trainer):

    absolute_start = time()
    device = trainer.device0

    dl_len = len(trainer.dataload.train)
    for batch_idx, sample in enumerate(trainer.dataload.train, start=1):

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

        ### do backprop
        trainer.optim.zero_grad()
        full_loss.backward()
        trainer.optim.step()

        trainer.meters.train.smpl_pose_loss.update(smpl_pose_loss.item(), n=batch_size)
        trainer.meters.train.smpl_shape_loss.update(
            smpl_shape_loss.item(), n=batch_size
        )
        trainer.meters.train.smpl_shape_prior_loss.update(
            smpl_shape_prior_loss.item(), n=batch_size
        )
        trainer.meters.train.kpts2d_loss.update(kpts2d_loss.item(), n=batch_size)
        trainer.meters.train.kpts3d_loss.update(kpts3d_loss.item(), n=batch_size)
        trainer.meters.train.camera_scale_reg_loss.update(
            camera_scale_reg_loss.item(), n=batch_size
        )
        trainer.meters.train.full.update(full_loss.item(), n=batch_size)

        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.full, total_time)


def valid(trainer):

    absolute_start = time()

    device = trainer.device0

    ### validate on the first batch of COCO dataset
    sample = next(iter(trainer.dataload.valid))
    batch_idx = 1
    img = sample["img"].to(device, non_blocking=True)  # Bx3x224x224
    gt_kpts2d_norm = sample["gt2d_norm"].to(device, non_blocking=True)  # Bx49x3
    batch_size = img.size(0)

    ### inference
    with torch.no_grad():
        pred_smpl_rotmat, pred_smpl_shape, pred_camera = trainer.models.hmrnet(img)
    pred_out = trainer.smpl_model_49(
        betas=pred_smpl_shape,
        body_pose=pred_smpl_rotmat[:, 1:],
        global_orient=pred_smpl_rotmat[:, :1],
        pose2rot=False,
    )

    pred_kpts3d = pred_out.joints
    pred_vertices = pred_out.vertices

    scale, trans = pred_camera[:, 0], pred_camera[:, 1:]
    pred_kpts2d_norm = weakProjection(pred_kpts3d, scale, trans)

    ### compute losses
    kpts2d_loss = trainer.losses.kpts2d_loss(pred_kpts2d_norm, gt_kpts2d_norm)
    trainer.meters.valid.kpts2d_loss.update(kpts2d_loss.item(), n=batch_size)

    ### visualize
    pred_vertices_bbox = convert_norm_points_to_bbox(
        pred_vertices.cpu(), scale.cpu(), trans.cpu(), sample["crop_shape"]
    )
    fig = plot_batch_with_mesh(
        img.cpu(), pred_vertices_bbox.cpu(), trainer.smpl_model_faces
    )

    local_fig = ospjoin(
        trainer.final_output_dir,
        f"fig_epoch{trainer.cur_epoch:05d}_batch{batch_idx:05d}.png",
    )
    fig.savefig(local_fig)
    manifold.copy_file_from_local(local_fig, trainer.remote_exp_dir)

    total_time = time() - absolute_start
    status_msg(trainer, batch_idx, 1, trainer.meters.valid.kpts2d_loss, total_time)

    ### evaluation on 3DPW
    pa_mpjpe = eval_on_3dpw(trainer)

    return pa_mpjpe
