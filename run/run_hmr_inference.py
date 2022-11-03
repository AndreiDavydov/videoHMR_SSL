import argparse
import os

import sys

import numpy as np

import torch

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from tqdm import tqdm

pathmgr = PathManager()
pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

_root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root_path)

import src
from src.datasets.datasets_common import NORMALIZE as Normalize

from src.functional import detector, smpl
from src.models import hmr
from src.utils.img_utils import convert_smpl_vertices_to_image_coord, crop_bbox
from src.utils.video_utils import get_frames, write_video
from src.utils.vis_utils import render_mesh_onto_image, vis_bbox, vis_kps


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hmr_path",
    required=False,
    default="manifold://xr_body/tree/personal/andreydavydov/eft/models_eft/2020_05_31-00_50_43-best-51.749683916568756.pt",
    help="Path to pretrained checkpoint",
)

parser.add_argument(
    "--smpl_model_path",
    required=False,
    default="manifold://xr_body/tree/personal/andreydavydov/eft/extradata/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
    help="Path to SMPL neutral body pkl file",
)

parser.add_argument(
    "--video_path",
    required=False,
    default="manifold://xr_body/tree/personal/andreydavydov/eft/sampledata/han_short.mp4",
    help="Path to input video to process",
)

parser.add_argument(
    "--save_video_path",
    required=False,
    default="/tmp/video.mp4",
    help="Path to save video with images and body prediction attached.",
)

parser.add_argument("--render", action="store_true", default=False)


def main():

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ### init smpl
    smpl_model = smpl.get_smpl_model("h36m", smpl_model_path=args.smpl_model_path).to(
        device
    )
    faces = smpl_model.faces.astype(int)

    ### init hmr
    hmr_model = hmr.hmr().to(device)
    ckpt = torch.load(pathmgr.get_local_path(args.hmr_path), map_location=device)
    hmr_model.load_state_dict(ckpt["model"], strict=False)
    hmr_model.eval()

    ### init bbox detector
    bbox_model = detector.init_detector()

    ### extract images from video
    frame_list, fps = get_frames(pathmgr.get_local_path(args.video_path))

    frames_out = []
    for idx in tqdm(range(len(frame_list))):
        img_original = frame_list[idx]

        ### bbox detection
        bbox, confidence, var, kpts = detector.inference(img_original, bbox_model)

        ### run inference img (cropped by bbox) -> smpl
        img, boxScale_o2n, bboxTopLeft = crop_bbox(img_original, bbox[0], out_size=224)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_norm = Normalize(img.clone())[None]

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = hmr_model(img_norm.to(device))
            pred_output = smpl_model(
                betas=pred_betas,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False,
            )
            pred_vertices = pred_output.vertices
            scale, trans = pred_camera[:, 0], pred_camera[:, 1:]

            boxScale_o2n = torch.tensor(boxScale_o2n)[None].to(device)
            bboxTopLeft = torch.tensor(bboxTopLeft)[None].to(device)
            img_orig_shape = torch.tensor(img_original.shape[:2])[None].to(device)
            img_crop_shape = torch.tensor((224, 224))[None].to(device)

            pred_vertices_img = convert_smpl_vertices_to_image_coord(
                pred_vertices,
                scale,
                trans,
                boxScale_o2n,
                bboxTopLeft,
                img_orig_shape,
                img_crop_shape,
            )[0]

        ### visualize

        if args.render:
            ### render with pytorch3d
            img = render_mesh_onto_image(
                img_original, pred_vertices_img.cpu().numpy(), faces
            )
        else:
            # mesh vertices
            img = vis_kps(img_original, pred_vertices_img)

        # # hmr kpts
        # img = vis_kps(img, pred_joints_img, size=5, color=(0, 0, 255))

        # # bbox detector kpts
        # img = vis_kps(img, kpts[0, :, :].permute(1, 0), size=5)
        # bbox detector bbox
        img = vis_bbox(img, bbox[0, :], format="xyxy")  # bbox

        frames_out.append(img)

    ### write a video
    write_video(frames_out, args.save_video_path, fps=10)


if __name__ == "__main__":
    main()
