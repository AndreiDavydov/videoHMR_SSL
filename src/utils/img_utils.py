# Taken from https://github.com/facebookresearch/eft/blob/main/eft/utils/imutils.py

"""
This file contains functions that are used to perform data augmentation.
"""

import cv2
import numpy as np
import scipy
import scipy.ndimage
import torch


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale  # h becomes the original bbox max(height, min).
    t = np.zeros((3, 3))
    t[0, 0] = (
        float(res[1]) / h
    )  # This becomes a scaling factor to rescale original bbox -> res size (default: 224x224)
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform([res[0], res[1]], center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if new_shape[0] > 15000 or new_shape[1] > 15000:
        print(
            "Image Size Too Big!  scale{}, new_shape{} br{}, ul{}".format(
                scale, new_shape, br, ul
            )
        )
        return None

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    if (
        new_y[1] - new_y[0] != old_y[1] - old_y[0]
        or new_x[1] - new_x[0] != old_x[1] - old_x[0]
        or new_y[1] - new_y[0] < 0
        or new_x[1] - new_x[0] < 0
    ):
        print("Warning: maybe person is out of image boundary!")
        return None
    new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]

    bboxScale_o2n = res[0] / new_img.shape[0]
    bboxTopLeft_inOriginal = (ul[0], ul[1])

    if not rot == 0:
        # Remove padding
        # new_img = scipy.misc.imrotate(new_img, rot) # most functions in misc are deprecated
        new_img = scipy.ndimage.interpolation.rotate(new_img, rot, reshape=False)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, tuple(res))

    return new_img, bboxScale_o2n, np.array(bboxTopLeft_inOriginal)


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img


def crop_bbox(img, bbox, out_size=224):
    """
    bbox : (4,) np.array, [ul_x, ul_y, br_x, br_y]
    """
    center = (bbox[:2] + bbox[2:]) / 2
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.0 * 1.2

    img, boxScale_o2n, bboxTopLeft = crop(
        img.copy(), center, scale, (out_size, out_size)
    )

    return img, boxScale_o2n, bboxTopLeft


def uncrop_points_bbox(points, boxScale_o2n, bboxTopLeft, orig_shape, crop_shape):
    """
    Converts the points from the coordinates cropped (by `crop` function) into
    the original size.
    points : B x N x dim (2 or 3)
    boxScale_o2n : B
    bboxTopLeft : B x 2
    orig_shape : B x 2
    crop_shape : B x 2
    """

    batch_size = points.size(0)
    boxScale_o2n = boxScale_o2n.view(batch_size, 1)
    points = points / boxScale_o2n.view(batch_size, 1, 1)

    points[:, :, :2] += bboxTopLeft.view(batch_size, 1, 2)
    return points


def convert_norm_points_to_bbox(points, scale, trans, img_size):
    """
    Scale and translate points from centered normalized space to bbox coordinates.
    points: B x N x dim (2 or 3)
    scale : B
    trans : B x 2
    img_size : B x 2 (here use only Height of the image for scaling)

    NOTE: output points are aligned with original (non-augmented) bbox
    To align with augmented, the same transformation must be applied to 3D points.
    """
    batch_size = points.size(0)
    points = points * scale.view(batch_size, 1, 1)
    points[:, :, 0:2] = points[:, :, 0:2] + trans.view(batch_size, 1, 2)
    height = img_size[:, 0].view(batch_size, 1, 1)
    points = points * height / 2
    points[:, :, 0:2] += height / 2  # assume  bbox H == bbox W (== 224)
    return points


def convert_smpl_vertices_to_image_coord(
    vertices,
    cam_scale,
    cam_trans,
    bbox_scale,
    bbox_top_left,
    orig_size,
    scaled_size,
):
    """
    vertices : B x N x 3
    cam_scale : B
    cam_trans : B x 2
    orig_size : B x 2
    scaled_size : B x 2
    """
    bbox_verts = convert_norm_points_to_bbox(
        vertices, cam_scale, cam_trans, scaled_size
    )

    img_verts = uncrop_points_bbox(
        bbox_verts, bbox_scale, bbox_top_left, orig_size, scaled_size
    )
    return img_verts


def weakProjection(points, scale, trans):
    batch_size = points.size(0)
    points = points * scale.view(batch_size, 1, 1)
    points = points[:, :, 0:2] + trans.view(batch_size, 1, 2)
    return points


def get_intrinsics_matrix(img_width, img_height, focal_length):
    """
    Camera intrinsic matrix (calibration matrix) given focal length in pixels and img_width and
    img_height. Assumes that principal point is at (width/2, height/2).
    """
    K = np.array(
        [
            [focal_length, 0.0, img_width / 2.0],
            [0.0, focal_length, img_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return K


def perspective_projection(
    points, rotation, translation, cam_K=None, focal_length=None, img_wh=None
):
    """
    This function computes the perspective projection of a set of points in torch.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        Either
        cam_K (bs, 3, 3): Camera intrinsics matrix
        Or
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    if cam_K is None:
        cam_K = torch.from_numpy(
            get_intrinsics_matrix(img_wh, img_wh, focal_length).astype(np.float32)
        )
        cam_K = torch.cat(batch_size * [cam_K[None, :, :]], dim=0)
        cam_K = cam_K.to(points.device)

    # Transform points
    if rotation is not None:
        points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    points_z = points[:, :, -1]
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", cam_K, projected_points)
    projected_points[:, :, 2] = points_z

    return projected_points
