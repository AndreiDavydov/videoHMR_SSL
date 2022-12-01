import os

import torch
import torch.nn as nn

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
from src.functional.tcmr import TemporalEncoder

from src.models.spin import Regressor

# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

BASE_DIR = "/cvlabdata2/home/davydov/videoHMR_SSL/dump_from_tcmr"
SPIN_FILE = "spin_model_checkpoint.pth.tar"
TCMR_DEMO_FILE = "tcmr_demo_model.pth.tar"
TCMR_TABLE4_FILE = "tcmr_table4_3dpw_test.pth.tar"


class TCMR(nn.Module):
    def __init__(
        self,
        seqlen,
        batch_size=64,
        n_layers=1,
        hidden_size=2048,
    ):
        super(TCMR, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            seq_len=seqlen, n_layers=n_layers, hidden_size=hidden_size
        )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        # pretrained = pathmgr.get_local_path(os.path.join(BASE_DIR, SPIN_FILE))
        pretrained = os.path.join(BASE_DIR, SPIN_FILE)
        pretrained_dict = torch.load(pretrained)["model"]

        self.regressor.load_state_dict(pretrained_dict, strict=False)
        print(f"=> loaded pretrained regressor model from '{pretrained}'")

    def forward(self, input, is_train=False, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        feature, scores = self.encoder(input, is_train=is_train)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(
            feature, is_train=is_train, J_regressor=J_regressor
        )

        if not is_train:
            for s in smpl_output:
                s["theta"] = s["theta"].reshape(batch_size, -1)
                s["verts"] = s["verts"].reshape(batch_size, -1, 3)
                s["kp_2d"] = s["kp_2d"].reshape(batch_size, -1, 2)
                s["kp_3d"] = s["kp_3d"].reshape(batch_size, -1, 3)
                s["rotmat"] = s["rotmat"].reshape(batch_size, -1, 3, 3)
                s["scores"] = scores

        else:
            repeat_num = 3
            for s in smpl_output:
                s["theta"] = s["theta"].reshape(batch_size, repeat_num, -1)
                s["verts"] = s["verts"].reshape(batch_size, repeat_num, -1, 3)
                s["kp_2d"] = s["kp_2d"].reshape(batch_size, repeat_num, -1, 2)
                s["kp_3d"] = s["kp_3d"].reshape(batch_size, repeat_num, -1, 3)
                s["rotmat"] = s["rotmat"].reshape(batch_size, repeat_num, -1, 3, 3)
                s["scores"] = scores

        return smpl_output, scores


def get_tcmr():
    ### get pretrained TCMR model that was used to reproduce results in Table 4
    # (see original TCMR paper)
    seqlen = 16
    model = TCMR(
        n_layers=2,
        batch_size=32,
        seqlen=seqlen,
        hidden_size=1024,
    )
    ckpt_path = os.path.join(BASE_DIR, TCMR_TABLE4_FILE)
    # checkpoint = torch.load(pathmgr.get_local_path(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["gen_state_dict"])
    print(f"==> Loaded pretrained model from {ckpt_path}...")

    model.eval()
    return model


if __name__ == "__main__":
    import numpy as np

    torch.set_grad_enabled(False)
    device = "cuda:0"

    from src.models.spin import hmr

    ### init feature extractor
    hmr = hmr().to(device)
    # pretrained = pathmgr.get_local_path(os.path.join(BASE_DIR, SPIN_FILE))
    pretrained = os.path.join(BASE_DIR, SPIN_FILE)
    pretrained_dict = torch.load(pretrained)["model"]
    hmr.load_state_dict(pretrained_dict, strict=False)
    hmr.eval()

    ### init TCMR with smpl regressor
    seqlen = 16
    tcmr_model = TCMR(seqlen=seqlen, n_layers=2, hidden_size=1024).to(device)
    # pretrained = pathmgr.get_local_path(os.path.join(BASE_DIR, TCMR_DEMO_FILE))
    pretrained = os.path.join(BASE_DIR, TCMR_DEMO_FILE)
    pretrained_dict = torch.load(pretrained)["gen_state_dict"]
    tcmr_model.load_state_dict(pretrained_dict, strict=False)
    tcmr_model.eval()

    # j_regr = pathmgr.get_local_path(
    #     "manifold://xr_body/tree/personal/andreydavydov/eft/extradata/data_from_spin/J_regressor_h36m.npy"
    # )
    j_regr = "/cvlabdata2/home/davydov/videoHMR_SSL/data/smpl_data/J_regressor_h36m.npy"
    J_regressor = torch.from_numpy(np.load(j_regr)).float()

    ### random inference image-features
    torch.manual_seed(0)
    image_batch = torch.randn((16, 3, 224, 224)).to(device)
    features = hmr.feature_extractor(image_batch)
    print(features.shape)

    ### random inference features-smpl
    output = tcmr_model(features.unsqueeze(0), J_regressor=J_regressor, is_train=False)
    output = output[0][-1]
    pred_cam = output["theta"][:, :3]
    pred_verts = output["verts"]
    pred_pose = output["theta"][:, 3:75]
    pred_betas = output["theta"][:, 75:]
    pred_joints3d = output["kp_3d"]
    print(pred_pose.shape, pred_pose.shape, pred_cam.shape)
    print(pred_verts.shape, pred_joints3d.shape)
