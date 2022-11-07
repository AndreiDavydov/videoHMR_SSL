from time import time

import torch
# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
from src.functional import smpl

from src.functional.optical_flow import (
    get_corners_for_points,
    unproject_corners_on_mesh,
    unproject_data_to_vertices,
)
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

        ### prepare verts3d for start/end mesh vertices
        verts3d_start, verts3d_end = verts3d[:-1], verts3d[1:]

        ### unproject OF onto M1
        unproj_flow2d_forward, vis_mask_start = unproject_data_to_vertices(
            verts3d[:-1], opt_flow_forward, trainer.smpl_model_faces, trainer.cameras
        )
        vis_mask_start = vis_mask_start.view(-1)

        ### compute displacements for Mesh1 vertices on Image2
        m1_on_i2 = verts3d_start[..., :2] + unproj_flow2d_forward  # (B-1) x N x 2

        px_quads, dx, dy = get_corners_for_points(m1_on_i2)
        (
            bary_coords_corners,
            verts_triplet_indices,
            vis_mask_end,
        ) = unproject_corners_on_mesh(
            px_quads,
            verts3d_end,
            trainer.smpl_model_faces,
            trainer.cameras,
            img_size=(img_size, img_size),
        )

        ### mask of verts of M1 that go to visible verts of M2
        vis_mask = (vis_mask_start * vis_mask_end).type(torch.bool)

        m1_vis_indices = torch.arange(len(vis_mask)).to(device)[vis_mask].view(-1, 1)
        m2_vis_indices_all = verts_triplet_indices.reshape(-1, 12)[vis_mask]
        m_map_to_same_indices = (m2_vis_indices_all == m1_vis_indices).type(torch.bool)

        bary_coords_corners = bary_coords_corners[vis_mask].reshape(-1, 12)

        print(
            m1_vis_indices.shape,
            m2_vis_indices_all.shape,
            m_map_to_same_indices.shape,
            bary_coords_corners.shape,
        )

        dx = dx.view(-1)[vis_mask]
        dy = dy.view(-1)[vis_mask]
        w_tl = dx * dy
        w_tr = (1 - dx) * dy
        w_bl = dx * (1 - dy)
        w_br = (1 - dx) * (1 - dy)
        w_faces = torch.stack((w_tl, w_tr, w_bl, w_br), dim=-1)
        w_faces = w_faces[..., None].repeat(1, 1, 3).view(-1, 12)

        w = bary_coords_corners * w_faces
        print(w.shape)

        loss = 0

        if trainer.losses_weights.flow_3d > 0:
            flow_3d = trainer.losses.flow_3d(w, m_map_to_same_indices)
            loss += flow_3d * trainer.losses_weights.flow_3d
            trainer.meters.train.flow_3d.update(flow_3d.item(), n=1)

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
