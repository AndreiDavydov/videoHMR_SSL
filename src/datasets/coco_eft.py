import json
import os
import shutil

import cv2
import numpy as np

import src.utils.augment as augment
import torch

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
from src.datasets.datasets_common import NORMALIZE


# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

COCO_EFT_IMG_FOLDER = "manifold://fair_vision_data/tree/coco_train2014_oct_15_2018/"
COCO_EFT_IMG_ZIP = (
    "manifold://xr_body/tree/personal/andreydavydov/eft/coco_train2014_oct_15_2018.zip"
)
COCO_EFT_IMG_FOLDER_LOCAL = "/tmp/coco_train/"
COCO_EFT_JSON_FILE = "manifold://xr_body/tree/personal/andreydavydov/eft/eft_fit/COCO2014-All-ver01.json"  # 74834 samples in COCO2014


def _load_kpts(kpts_path):
    # if kpts_path.startswith("manifold://"):
    #     kpts_path = pathmgr.get_local_path(kpts_path)

    with open(kpts_path, "r") as f:
        kpts = json.load(f)
    return kpts


### structure of kpts json:
# > json keys: 'ver', 'data', 'meta'
# > json["data"]: list of datasamples
# its fields:
# - 'parm_pose' (24,3,3) - SMPL pose parameters
# - 'parm_shape' (10,) - SMPL shape parameters
# - 'parm_cam' (3,) SMPL global orientation
# - 'bbox_scale' - bbox scale, scalar
# - 'bbox_center' - bbox center, [cx, cy]
# - 'gt_keypoint_2d' (49,3) - image keypoints (3rd dim is mask)
# - 'joint_validity_openpose18' (18,) binary mask of visibility of openpose keypoints
# - 'smpltype' - "smpl"
# - 'annotId' - some unique id
# - 'imageName' - image filename for this sample, e.g. "COCO_train2014_000000000036.jpg"


class COCO_EFT(torch.utils.data.Dataset):
    def __init__(
        self,
        is_train=True,
        do_scale=True,
        do_rot=True,
        do_flip=True,
        do_noise=True,
        copy_to_local=True,
    ):

        # if copy_to_local:
        #     ### copy zip to local
        #     images_zip = pathmgr.get_local_path(COCO_EFT_IMG_ZIP)
        #     ### unarchive
        #     image_files_dir = os.path.join(
        #         COCO_EFT_IMG_FOLDER_LOCAL, "coco_train2014_oct_15_2018"
        #     )
        #     if not os.path.exists(image_files_dir):
        #         os.makedirs(COCO_EFT_IMG_FOLDER_LOCAL)
        #         shutil.unpack_archive(images_zip, COCO_EFT_IMG_FOLDER_LOCAL)
        # else:
        #     image_files_dir = COCO_EFT_IMG_FOLDER

        image_files_dir = COCO_EFT_IMG_FOLDER

        self.kpts = _load_kpts(COCO_EFT_JSON_FILE)["data"]
        self.img_folder = image_files_dir

        self.is_train = is_train

        self.do_scale = do_scale
        self.do_rot = do_rot
        self.do_flip = do_flip
        self.do_noise = do_noise

        self.normalize_img = NORMALIZE

    def __len__(self):
        return len(self.kpts)

    def get_img(self, imageName):
        # img_path = pathmgr.get_local_path(os.path.join(self.img_folder, imageName))
        img_path = os.path.join(self.img_folder, imageName)
        img = cv2.imread(img_path)[:, :, ::-1]
        return img

    def __getitem__(self, idx):
        sample = self.kpts[idx]

        ### augmentation: scaling, rotation, flipping, noise
        flip, pn, rot, sc = augment.augm_params(
            self.is_train, self.do_scale, self.do_rot, self.do_flip, self.do_noise
        )

        ### image processing: crop, rescale, rotate, flip, noise
        scale, center = sample["bbox_scale"], sample["bbox_center"]
        img_original = self.get_img(sample["imageName"])
        img = img_original.copy().astype(np.float32)
        orig_shape = np.array(img.shape)[:2]
        img, bboxScale_o2n, bboxTopLeft = augment.rgb_augmentation(
            img, center, sc * scale, rot, flip, pn
        )
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)
        crop_shape = img.shape[-2:]

        ### SMPL data processing
        smpl_pose = np.array(sample["parm_pose"])  # (24,3,3) - rot matrix
        smpl_pose = augment.pose_augment(smpl_pose, rot, flip)
        smpl_pose = torch.from_numpy(smpl_pose).float()

        smpl_shape = np.array(sample["parm_shape"])  # (10,)
        smpl_shape = torch.from_numpy(smpl_shape).float()

        ### 2d keypoints processing
        gt2d = np.array(sample["gt_keypoint_2d"])  # (49,3)
        imgHeight = orig_shape[0]
        if (
            abs(gt2d[25 + 0, 1] - imgHeight) < 10 and gt2d[10, 2] < 0.1
        ):  # Right Foot. within 10 pix from the boundary
            gt2d[25 + 0, 2] = 0  # Disable
        if abs(gt2d[25 + 5, 1] - imgHeight) < 10 and gt2d[13, 2] < 0.1:  # Left Foot.
            gt2d[25 + 5, 2] = 0  # Disable

        gt2d = augment.j2d_augment(gt2d.copy(), center, sc * scale, rot, flip)
        gt2d = torch.from_numpy(gt2d).float()

        ### normalize 2d keypoints for training from -1 to 1
        gt2d_norm = gt2d.clone()
        gt2d_norm[:, :-1] = 2.0 * gt2d[:, :-1] / augment.IMG_RES - 1.0

        out = {
            "img": img,
            "gt2d": gt2d,
            "gt2d_norm": gt2d_norm,
            "gt2d_original": torch.tensor(sample["gt_keypoint_2d"]),
            "smpl_pose": smpl_pose,
            "smpl_shape": smpl_shape,
            "imgName": sample["imageName"],
            "bbox_scale": torch.tensor(sample["bbox_scale"]),
            "bbox_center": torch.tensor(sample["bbox_center"]),
            "augmentation": {
                "flip": flip,
                "center": center,
                "scale": sc * scale,
                "rot": rot,
            },
            "parm_cam": torch.tensor(sample["parm_cam"]),
            "bboxInfoScale": torch.tensor(bboxScale_o2n),
            "bboxInfoTopLeft": torch.tensor(bboxTopLeft),
            "orig_shape": torch.tensor(orig_shape),
            "crop_shape": torch.tensor(crop_shape),
        }

        return out


if __name__ == "__main__":
    dset = COCO_EFT()
    print(len(dset))
    print(dset[0].keys())
