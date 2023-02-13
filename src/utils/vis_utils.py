import copy

import cv2
import numpy as np
import torch

from pytorch3d.renderer import (
    BlendParams,
    FoVOrthographicCameras,
    MeshRasterizer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    Textures,
)
from pytorch3d.structures import Meshes


def vis_bbox(img, bbox, color=(0, 255, 0), bbox_thickness=3, format="xywh"):
    bbox = [int(x + 0.5) for x in bbox]  # [x, y, w, h]
    img_bbox = copy.deepcopy(img)

    if format == "xywh":
        img_bbox = cv2.rectangle(
            img_bbox,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            color=color,
            thickness=bbox_thickness,
        )
    else:
        img_bbox = cv2.rectangle(
            img_bbox,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color=color,
            thickness=bbox_thickness,
        )

    return img_bbox


def vis_kps(img, kps, color=(0, 255, 0), size=1):
    img_kp = copy.deepcopy(img)
    for kp in kps:
        img_kp = cv2.drawMarker(
            img_kp,
            (int(kp[0]), int(kp[1])),
            color=color,
            markerType=cv2.MARKER_STAR,
            thickness=1,
            markerSize=size,
        )
    return img_kp


def render_mesh_onto_image(
    img, vertices, faces, device="cuda", rasterizer=None, shader=None
):
    """
    img : H x W x 3 np.array
    vertices : N x 3 np.array - mesh vertices (aligned with object in the image)
    faces : M x 3 np.array - list of indices of vertex triplets
    """

    ### map vertices to ndc format
    vertices = vertices.copy()
    vertices[:, 0] -= img.shape[1] / 2
    vertices[:, 1] -= img.shape[0] / 2

    vertices[:, 0:2] *= -1

    min_size = min(img.shape[:2])
    vertices /= min_size / 2
    vertices[:, 2] += 10

    if rasterizer is None or shader is None:

        img_shape = img.shape[:2]

        cameras = FoVOrthographicCameras(device=device, zfar=100)

        raster_settings = RasterizationSettings(
            image_size=[img_shape[0], img_shape[1]],
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
        blend_params = BlendParams(background_color=(0, 0, 0))

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params,
        )

    pytorch_mesh = convert_to_pytorch_mesh(vertices, faces, device=device)

    fragments = rasterizer(pytorch_mesh)
    images = shader(fragments, pytorch_mesh)
    render_img = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

    # blend with original image
    alpha_mask = (fragments.pix_to_face.cpu().squeeze().numpy() != -1).astype(np.int)
    alpha_mask = np.tile(alpha_mask[:, :, np.newaxis], (1, 1, 3))
    render_img = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

    blend_img = (alpha_mask * render_img + (1 - alpha_mask) * img).astype(np.uint8)

    return blend_img


def render_verts_only(verts, faces, device, img_size=512):
    verts_mean = verts.copy().mean(axis=0, keepdims=True)
    verts -= verts_mean
    scale = 1  # 0.8
    verts *= scale
    verts = (verts + 1) / 2 * img_size
    img = (255 * np.ones((img_size, img_size, 3))).astype("uint8")
    blend = render_mesh_onto_image(img, verts, faces, device)
    return blend


def convert_to_pytorch_mesh(mesh, faces, verts_rgb_colors=None, device="cuda"):

    if verts_rgb_colors is None:
        verts_rgb_colors = 0.6 * torch.ones([1, mesh.shape[0], 3]).to(device)
        verts_rgb_colors[..., 0] += 0.4
    tex = Textures(verts_rgb=verts_rgb_colors)

    pytorch_mesh = Meshes(
        verts=torch.Tensor(mesh).unsqueeze(0),
        faces=torch.Tensor(faces).unsqueeze(0),
        textures=tex,
    ).to(device)

    return pytorch_mesh


def render_mesh_onto_image_batch(
    img, vertices, faces, device="cuda", rasterizer=None, shader=None, verts_rgb_colors=None
):
    """
    img : B x 3 x H x W torch.tensor
    vertices : B x N x 3 torch.tensor - mesh vertices (aligned with object in the image)
    faces : B x M x 3 torch.tensor - list of indices of vertex triplets
    """

    ### map vertices to ndc format
    H, W = img.shape[-2], img.shape[-1]
    vertices = vertices.clone()
    vertices[:, :, 0] -= H / 2
    vertices[:, :, 1] -= W / 2

    vertices[:, :, 0:2] *= -1

    min_size = min(H, W)
    vertices /= min_size / 2
    vertices[:, :, 2] += 10

    if rasterizer is None or shader is None:
        cameras = FoVOrthographicCameras(device=device, zfar=100)
        raster_settings = RasterizationSettings(
            image_size=[H, W],
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
        blend_params = BlendParams(background_color=(0, 0, 0))

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params,
        )

    if verts_rgb_colors is None:
        verts_rgb_colors = 0.6 * torch.ones_like(vertices)
        verts_rgb_colors[..., 0] += 0.4
    tex = Textures(verts_rgb=verts_rgb_colors)

    pytorch_mesh = Meshes(
        verts=torch.Tensor(vertices),
        faces=torch.Tensor(faces),
        textures=tex,
    ).to(device)

    fragments = rasterizer(pytorch_mesh)
    images = shader(fragments, pytorch_mesh)

    render_img = images[..., :3]
    img = img.permute(0,2,3,1).to(device) / 255
    # blend with original image
    alpha_mask = (fragments.pix_to_face != -1).float()
    blend_img = render_img * alpha_mask + img * (1 - alpha_mask)

    return blend_img.cpu()