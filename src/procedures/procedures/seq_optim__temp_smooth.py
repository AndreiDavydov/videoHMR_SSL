from time import time

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler

from src.functional import smpl
from src.procedures.procedures_common import status_msg

# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)


def setup(trainer):

    ### init smpl
    trainer.smpl_model_14 = smpl.get_smpl_model("h36m", device=trainer.device0)
    trainer.smpl_model_49 = smpl.get_smpl_model("extra", device=trainer.device0)


def train(trainer):
    absolute_start = time()
    for iteration in range(1, trainer.cfg.TRAINING.NUM_ITERATIONS + 1):
        trainer.optim.zero_grad()
        j2d_cur, j3d_cur, verts_cur = trainer.models.seqOpt(trainer.smpl_model_49)
        j2d_init, j3d_init, verts_init = trainer.models.seqOpt.module.forward_with_init(
            trainer.smpl_model_49
        )
        loss = 0

        ### shape_smoothing
        if trainer.losses_weights.shape_smooth > 0:
            shape_init = trainer.models.seqOpt.module.shape_init
            shape_cur = trainer.models.seqOpt.module.shape
            shape_smooth = trainer.losses.shape_smooth(shape_cur, shape_init)
            loss += shape_smooth * trainer.losses_weights.shape_smooth

        ### pose 3d smoothing
        if trainer.losses_weights.j3d_smooth > 0:
            j3d_smooth = trainer.losses.shape_smooth(j3d_cur, j3d_init)
            loss += j3d_smooth * trainer.losses_weights.j3d_smooth

        ### verts smoothing
        if trainer.losses_weights.verts_smooth > 0:
            verts_smooth = trainer.losses.shape_smooth(verts_cur, verts_init)
            loss += verts_smooth * trainer.losses_weights.verts_smooth

        loss.backward()

        trainer.optim.step()

        trainer.meters.train.full.update(loss.item(), n=1)
        total_time = time() - absolute_start
        status_msg(
            trainer,
            iteration,
            trainer.cfg.TRAINING.NUM_ITERATIONS,
            trainer.meters.train.full,
            total_time,
        )


def valid(trainer):
    absolute_start = time()
    device = trainer.device0

    ### prepare the result of optimization
    _, j3d_pred, verts_pred = trainer.models.seqOpt(trainer.smpl_model_14)
    j2d_pred, _, _ = trainer.models.seqOpt(trainer.smpl_model_49)

    ### prepare initial prediction
    _, j3d_0, verts_0 = trainer.models.seqOpt.module.forward_with_init(
        trainer.smpl_model_14
    )

    ### read gt
    theta = trainer.models.seqOpt.module.orig_seq["theta"].to(device)
    gt_smpl_pose, gt_smpl_shape = theta[:, 3:75], theta[:, 75:]

    gt_out = trainer.smpl_model_14(
        betas=gt_smpl_shape,
        body_pose=gt_smpl_pose[:, 3:],
        global_orient=gt_smpl_pose[:, :3],
    )
    verts_gt = gt_out.vertices

    # 49 joints (normalized)
    j2d_gt = trainer.models.seqOpt.module.orig_seq["kp_2d"].to(device)
    # 14 joints
    j3d_gt = trainer.models.seqOpt.module.orig_seq["kp_3d"].to(device)

    ### compute losses and log
    pose2d_loss = trainer.losses.pose2d(j2d_pred, j2d_gt).mean()

    m2mm = 1000
    mpjpe_pred_vs_gt = trainer.losses.mpjpe_vs_gt(j3d_pred, j3d_gt).mean() * m2mm
    mpjpe_pred_vs_0 = trainer.losses.mpjpe_vs_0(j3d_pred, j3d_0).mean() * m2mm

    pa_mpjpe_pred_vs_gt = trainer.losses.pa_mpjpe_vs_gt(j3d_pred, j3d_gt).mean() * m2mm
    pa_mpjpe_pred_vs_0 = trainer.losses.pa_mpjpe_vs_0(j3d_pred, j3d_0).mean() * m2mm

    mpvpe_pred_vs_gt = trainer.losses.mpvpe_vs_gt(verts_pred, verts_gt).mean() * m2mm
    mpvpe_pred_vs_0 = trainer.losses.mpvpe_vs_0(verts_pred, verts_0).mean() * m2mm

    pa_mpvpe_pred_vs_gt = (
        trainer.losses.pa_mpvpe_vs_gt(verts_pred, verts_gt).mean() * m2mm
    )
    pa_mpvpe_pred_vs_0 = trainer.losses.pa_mpvpe_vs_0(verts_pred, verts_0).mean() * m2mm

    accel_pred = trainer.losses.accel(j3d_pred).mean() * m2mm
    accel_gt = trainer.losses.accel(j3d_gt).mean() * m2mm

    accel_err_pred_vs_gt = (
        trainer.losses.accel_err_vs_gt(j3d_pred.detach(), j3d_gt.detach()).mean() * m2mm
    )
    accel_err_pred_vs_0 = (
        trainer.losses.accel_err_vs_0(j3d_pred.detach(), j3d_0).mean() * m2mm
    )

    ### tracking
    trainer.meters.valid.pose2d.update(pose2d_loss.item(), n=1)

    trainer.meters.valid.mpjpe_vs_gt.update(mpjpe_pred_vs_gt.item(), n=1)
    trainer.meters.valid.mpjpe_vs_0.update(mpjpe_pred_vs_0.item(), n=1)

    trainer.meters.valid.pa_mpjpe_vs_gt.update(pa_mpjpe_pred_vs_gt.item(), n=1)
    trainer.meters.valid.pa_mpjpe_vs_0.update(pa_mpjpe_pred_vs_0.item(), n=1)

    trainer.meters.valid.mpvpe_vs_gt.update(mpvpe_pred_vs_gt.item(), n=1)
    trainer.meters.valid.mpvpe_vs_0.update(mpvpe_pred_vs_0.item(), n=1)

    trainer.meters.valid.pa_mpvpe_vs_gt.update(pa_mpvpe_pred_vs_gt.item(), n=1)
    trainer.meters.valid.pa_mpvpe_vs_0.update(pa_mpvpe_pred_vs_0.item(), n=1)

    trainer.meters.valid.accel_err_vs_gt.update(accel_err_pred_vs_gt.item(), n=1)
    trainer.meters.valid.accel_err_vs_0.update(accel_err_pred_vs_0.item(), n=1)

    ### logging
    trainer.logger.info(f"2D alignment: {pose2d_loss:5.2f} px")
    trainer.logger.info(
        f"MPJPE vs GT: {mpjpe_pred_vs_gt:5.2f} mm | vs Init: {mpjpe_pred_vs_0:5.2f} mm",
    )
    trainer.logger.info(
        f"PA-MPJPE vs GT: {pa_mpjpe_pred_vs_gt:5.2f} mm | vs Init: {pa_mpjpe_pred_vs_0:5.2f} mm",
    )
    trainer.logger.info(
        f"MPVPE vs GT: {mpvpe_pred_vs_gt:5.2f} mm | vs Init: {mpvpe_pred_vs_0:5.2f} mm",
    )

    trainer.logger.info(
        f"PA-MPVPE vs GT: {pa_mpvpe_pred_vs_gt:5.2f} mm | vs Init: {pa_mpvpe_pred_vs_0:5.2f} mm",
    )
    trainer.logger.info(f"Accel pred: {accel_pred:5.2f} mm/s^2")
    trainer.logger.info(f"Accel gt: {accel_gt:5.2f} mm/s^2")

    trainer.logger.info(
        f"Accel Err vs GT: {accel_err_pred_vs_gt:5.2f} mm/s^2 | vs Init: {accel_err_pred_vs_0:5.2f} mm/s^2",
    )

    trainer.logger.info(f"Total time: {time()-absolute_start:5.1f}s")

    return pa_mpjpe_pred_vs_gt
