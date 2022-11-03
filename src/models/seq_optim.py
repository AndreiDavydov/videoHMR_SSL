import torch
import torch.nn as nn

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from kornia.geometry import rotation_matrix_to_angle_axis as mat_to_aa
from src.utils.img_utils import weakProjection

pathmgr = PathManager()
pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)


def get_smpl_out(shape, pose, camera, smpl_model, pose2rot=True):
    out = smpl_model(
        betas=shape,
        body_pose=pose[:, 1:],
        global_orient=pose[:, :1],
        pose2rot=pose2rot,
    )
    verts = out.vertices
    j3d = out.joints

    j2d = weakProjection(j3d, camera[:, 0], camera[:, 1:])
    return j2d, j3d, verts


class SeqModel(torch.nn.Module):
    def __init__(self, seq_path=None, seq_num_frames=None, optical_flow_presaved=None):
        super(SeqModel, self).__init__()

        self.seq_path = seq_path
        self.seq_num_frames = seq_num_frames

        if seq_path is not None:
            self.init_seq(seq_path, seq_num_frames)

        if optical_flow_presaved is not None:
            self.optical_flow = torch.load(
                pathmgr.get_local_path(optical_flow_presaved)
            )

    def init_seq(self, f, num):
        seq = torch.load(pathmgr.get_local_path(f))
        for k in seq:
            seq[k] = seq[k][:num]  # keep only num frames
        ### keys of the sequence:
        # ['img', 'theta', 'kp_2d', 'kp_3d', 'pred_rotmat', 'pred_shape', 'pred_camera', 'instance_id']

        self.orig_seq = seq

        ### init seq params to optimize
        rotmat_init = seq["pred_rotmat"]
        shape_init = seq["pred_shape"]
        camera_init = seq["pred_camera"]

        pose_init = mat_to_aa(rotmat_init)  # .contiguous()
        self.pose = nn.Parameter(pose_init)  # it should be better to optimize in AA
        self.shape = nn.Parameter(shape_init)
        self.camera = nn.Parameter(camera_init)

        ### init all tensors before optimization
        self.register_buffer("shape_init", shape_init.detach().clone())
        self.register_buffer("pose_init", pose_init.detach().clone())
        self.register_buffer("camera_init", camera_init.detach().clone())

    def forward_with_init(self, smpl_model):
        j2d, j3d, verts = get_smpl_out(
            self.shape_init, self.pose_init, self.camera_init, smpl_model, pose2rot=True
        )
        return j2d, j3d, verts

    def forward(self, smpl_model):
        j2d, j3d, verts = get_smpl_out(
            self.shape, self.pose, self.camera, smpl_model, pose2rot=True
        )
        return j2d, j3d, verts


if __name__ == "__main__":

    seq = "manifold://xr_body/tree/personal/andreydavydov/3dpw_seq_for_tests/sample_3dpw_test_seq_downtown_runForBus_01_0__hmr_coco_all.pth"
    num = 123
    device = "cuda"
    optical_flow_presaved = "manifold://xr_body/tree/personal/andreydavydov/3dpw_seq_for_tests/optical_flow_presaved/optical_flows_for_seq.pth"

    seqOpt = SeqModel(seq, num, optical_flow_presaved=optical_flow_presaved).to(device)

    from src.functional import smpl

    smpl_model = smpl.get_smpl_model("h36m", device=device)

    j2d, j3d, verts = seqOpt(smpl_model)
    print(j2d.shape)
    print(j3d.shape)
    print(verts.shape)

    print("Optical Flow from RAFT, presaved:")
    print(seqOpt.optical_flow["forward_time"].shape)
    print(seqOpt.optical_flow["backward_time"].shape)
