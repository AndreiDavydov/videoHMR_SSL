# adapted from https://github.com/facebookresearch/eft/blob/main/eft/datasets/base_dataset.py

import cv2
import numpy as np
import torch
from kornia.geometry import rotation_matrix_to_angle_axis as mat_to_aa

from src.utils import img_utils

IMG_RES = 224


def augm_params(
    is_train,
    do_scale=True,
    do_rot=True,
    do_flip=True,
    do_noise=True,
    noise_factor=0.4,
    rot_factor=30,
    scale_factor=0.25,
):
    """augmentation parameters."""
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
    if is_train:
        # We flip with probability 1/2
        if do_flip and np.random.uniform() <= 0.5:
            flip = 1

        if do_noise:
            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

        if do_rot:
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(
                2 * rot_factor, max(-2 * rot_factor, np.random.randn() * rot_factor)
            )
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        if do_scale:
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(
                1 + scale_factor,
                max(1 - scale_factor, np.random.randn() * scale_factor + 1),
            )

    return flip, pn, rot, sc


def rgb_augmentation(
    rgb_img, center, scale, rot, flip, pn, img_size=(IMG_RES, IMG_RES)
):
    ### crop image
    rgb_img, bboxScale_o2n, bboxTopLeft = img_utils.crop(
        rgb_img, center, scale, img_size, rot
    )

    ### flip image
    if flip:
        rgb_img = img_utils.flip_img(rgb_img)

    ### add noise
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))

    # (3,IMG_RES,IMG_RES), float, [0,1]
    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
    return rgb_img, bboxScale_o2n, bboxTopLeft


# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    9,
    11,
    10,
    12,
    14,
    13,
    15,
    17,
    16,
    19,
    18,
    21,
    20,
    23,
    22,
]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def flip_pose_aa(pose_aa):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = SMPL_POSE_FLIP_PERM
    pose_aa = pose_aa[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose_aa[1::3] = -pose_aa[1::3]
    pose_aa[2::3] = -pose_aa[2::3]
    return pose_aa


def pose_augment(pose_rotmat, rot, flip):
    """Process SMPL theta parameters and apply all augmentation transforms."""
    J = pose_rotmat.shape[0]

    ### transform to axis-angle
    pose_rotmat = torch.from_numpy(pose_rotmat)
    pose_aa = mat_to_aa(pose_rotmat).contiguous().view(J * 3).numpy()

    ### rotation of the pose parameters
    pose_aa[:3] = rot_aa(pose_aa[:3], rot)

    ### flip the pose parameters
    if flip:
        pose_aa = flip_pose_aa(pose_aa)

    pose_aa = pose_aa.astype("float32")
    return pose_aa


# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [
    5,
    4,
    3,
    2,
    1,
    0,
    11,
    10,
    9,
    8,
    7,
    6,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    20,
    23,
    22,
]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [
    0,
    1,
    5,
    6,
    7,
    2,
    3,
    4,
    8,
    12,
    13,
    14,
    9,
    10,
    11,
    16,
    15,
    18,
    17,
    22,
    23,
    24,
    19,
    20,
    21,
] + [25 + i for i in J24_FLIP_PERM]


def flip_kp(kp, img_width=IMG_RES):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:, 0] = img_width - 1 - kp[:, 0]
    return kp


def j2d_augment(pose2d, center, scale, rot=0, flip=False, img_size=(IMG_RES, IMG_RES)):
    """Process gt 2D keypoints and apply all augmentation transforms."""

    ### scale and rotate
    nparts = pose2d.shape[0]
    for i in range(nparts):
        pose2d[i, 0:2] = img_utils.transform(
            pose2d[i, 0:2], center, scale, img_size, rot=rot
        )

    ### flip
    if flip:
        pose2d = flip_kp(pose2d)

    pose2d = pose2d.astype("float32")
    return pose2d
