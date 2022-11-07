from time import time

import torch
# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
from src.functional import smpl
from src.functional.optical_flow import unproject_optical_flows_to_vertices
from src.functional.renderer import get_default_cameras

from src.procedures.procedures.seq_optim__temp_smooth import valid as valid_on_seq
from src.procedures.procedures_common import status_msg


# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)


def setup(trainer):
    device = trainer.device0

    ### init smpl
    trainer.smpl_model_14 = smpl.get_smpl_model("h36m", device=device)
    trainer.smpl_model_49 = smpl.get_smpl_model("extra", device=device)

    ### set faces for renderer
    smpl_model_faces = trainer.smpl_model_49.faces.astype(int)
    trainer.smpl_model_faces = torch.tensor(smpl_model_faces.copy()).unsqueeze(0)

    ### init cameras
    trainer.cameras = get_default_cameras(device, mode="orthographic")


def train(trainer):
    absolute_start = time()
    device = trainer.device0
    opt_flow_forward = trainer.models.seqOpt.module.optical_flow["forward_time"]
    opt_flow_backward = trainer.models.seqOpt.module.optical_flow["backward_time"]
    opt_flow_forward = opt_flow_forward.to(device)
    opt_flow_backward = opt_flow_backward.to(device)

    img_size = opt_flow_forward.size(-1)

    for iteration in range(1, trainer.cfg.TRAINING.NUM_ITERATIONS + 1):

        trainer.optim.zero_grad()

        _, _, verts3d = trainer.models.seqOpt(trainer.smpl_model_49)
        batch_size = verts3d.size(0)

        ### align vertices with pixels
        camera = trainer.models.seqOpt.module.camera
        scale, trans = camera[:, 0].to(device), camera[:, 1:].to(device)
        verts3d = verts3d * scale.view(batch_size, 1, 1)
        verts3d[:, :, 0:2] = verts3d[:, :, 0:2] + trans.view(batch_size, 1, 2)
        verts3d = (verts3d + 1) / 2 * img_size

        verts_flow2d_forward = verts3d[1:, :, :2] - verts3d[:-1, :, :2]
        verts_flow2d_backward = verts3d[:-1, :, :2] - verts3d[1:, :, :2]

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

        ### Flow 3d (2d flow unprojection)
        if trainer.losses_weights.flow_3d > 0:
            # verts3d_flow = verts3d[1:] - verts3d[:-1]  # B-1 x Nverts x 3
            # flow_3d = trainer.losses.flow_3d(verts3d_flow, optical_flow[:1])
            # trainer.meters.train.flow_3d.update(flow_3d.item(), n=1)
            raise NotImplementedError  # TODO

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
    return valid_on_seq(trainer)
