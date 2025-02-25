from time import time

import torch

from src.functional import smpl
from tqdm import tqdm


def setup(trainer):
    ### init smpl
    if "SMPL" in trainer.cfg:
        j_regr = trainer.cfg.SMPL.j_regr
        batch_size = trainer.cfg.DATALOAD.eval.PARAMS.batch_size
        smpl_model = smpl.get_smpl_model(j_regr, batch_size, device=trainer.device0)
        trainer.smpl_model = smpl_model


def prepare_pred_gt(trainer, evaluation_accumulators):
    device = trainer.device0

    for sample in tqdm(trainer.dataload.eval):

        ### precompute gt
        theta = (
            sample["theta"].view(-1, 85).to(device, non_blocking=True)
        )  # B*seqlen x 85
        gt_smpl_pose, gt_smpl_shape = theta[:, 3:75], theta[:, 75:]

        gt_out = trainer.smpl_model(
            betas=gt_smpl_shape,
            body_pose=gt_smpl_pose[:, 3:],
            global_orient=gt_smpl_pose[:, :3],
        )
        gt_vertices = gt_out.vertices  # B*seqlen x 6890 x 3
        gt_kpts_3d = sample["kp_3d"]  # B x seqlen x J=14 x 3
        gt_kpts_3d = gt_kpts_3d.view(-1, gt_kpts_3d.size(2), 3)  # B*seqlen x J=14 x 3

        ### inference
        img = sample["video"].to(
            device, non_blocking=True
        )  # B x seqlen x 3 x 224 x 224
        img = img.view(-1, 3, 224, 224)  # B*seqlen x 3 x 224 x 224
        with torch.no_grad():
            pred_smpl_rotmat, pred_smpl_shape, pred_camera = trainer.models.hmrnet(img)
        pred_out = trainer.smpl_model(
            betas=pred_smpl_shape,
            body_pose=pred_smpl_rotmat[:, 1:],
            global_orient=pred_smpl_rotmat[:, :1],
            pose2rot=False,
        )
        pred_vertices = pred_out.vertices  # B*seqlen x 6890 x 3
        pred_kpts3d = pred_out.joints  # B*seqlen x J=14 x 3

        ### update accumulators
        evaluation_accumulators["pred_verts"].append(pred_vertices.cpu())
        evaluation_accumulators["gt_verts"].append(gt_vertices.cpu())
        evaluation_accumulators["pred_j3d"].append(pred_kpts3d.cpu())
        evaluation_accumulators["gt_j3d"].append(gt_kpts_3d.cpu())

    for k, v in evaluation_accumulators.items():
        evaluation_accumulators[k] = torch.cat(v)

    return evaluation_accumulators


def valid(trainer):

    absolute_start = time()

    ### init metrics
    evaluation_accumulators = {
        k: [] for k in ["pred_j3d", "gt_j3d", "pred_verts", "gt_verts"]
    }

    ### validate
    trainer.logger.info("Preparing tensors (validate)...")
    evaluation_accumulators = prepare_pred_gt(trainer, evaluation_accumulators)

    for k, v in evaluation_accumulators.items():
        trainer.logger.info(f"{k} {v.shape}")
    trainer.logger.info(f"Total time: {time()-absolute_start:5.1f}s")

    ### evaluate
    trainer.logger.info("Evaluating...")
    m2mm = 1000

    ### 3D keypoints
    pred_j3ds = evaluation_accumulators["pred_j3d"].clone()
    gt_j3ds = evaluation_accumulators["gt_j3d"].clone()

    mpjpe = trainer.losses.mpjpe(pred_j3ds, gt_j3ds).mean() * m2mm
    pa_mpjpe = trainer.losses.pa_mpjpe(pred_j3ds, gt_j3ds).mean() * m2mm

    ### Mesh vertices
    pred_verts = evaluation_accumulators["pred_verts"].clone()
    gt_verts = evaluation_accumulators["gt_verts"].clone()
    mpvpe = trainer.losses.mpvpe(pred_verts, gt_verts).mean() * m2mm

    ### acceleration
    pred_j3ds = evaluation_accumulators["pred_j3d"].clone()
    gt_j3ds = evaluation_accumulators["gt_j3d"].clone()
    accel_pred = trainer.losses.accel(pred_j3ds).mean() * m2mm
    accel_gt = trainer.losses.accel(gt_j3ds).mean() * m2mm
    accel_err = trainer.losses.accel_err(pred_j3ds, gt_j3ds).mean() * m2mm

    trainer.logger.info(f"MPJPE: {mpjpe:5.2f} mm")
    trainer.logger.info(f"PA-MPJPE: {pa_mpjpe:5.2f} mm")
    trainer.logger.info(f"MPVPE: {mpvpe:5.2f} mm")
    trainer.logger.info(f"Accel pred: {accel_pred:5.2f} mm/s^2")
    trainer.logger.info(f"Accel gt: {accel_gt:5.2f} mm/s^2")
    trainer.logger.info(f"Accel Err: {accel_err:5.2f} mm/s^2")
    trainer.logger.info(f"Total time: {time()-absolute_start:5.1f}s")

    trainer.meters.valid.mpjpe.update(mpjpe, n=1)
    trainer.meters.valid.pa_mpjpe.update(pa_mpjpe, n=1)
    trainer.meters.valid.mpvpe.update(mpvpe, n=1)
    trainer.meters.valid.accel.update(accel_pred, n=1)
    trainer.meters.valid.accel_err.update(accel_err, n=1)

    torch.save(pred_verts, trainer.cfg.SAVE_OUT)

    return pa_mpjpe
