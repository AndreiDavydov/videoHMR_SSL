import pytorch3d.renderer
import torch
from src.functional.renderer import (
    convert_vertices_to_mesh,
    fit_vertices_to_orthographic,
    get_vertex_visibility_mask,
    unproject_to_vertices,
)

def unproject_texture_to_vertices(verts3d, img, faces, cameras):
    """
    Args:
        verts3d (torch.tensor) B x N x 3 - mesh vertices, start/end together.
            X,Y coordinates are in pixels.
        img (torch.tensor) B x 3 x H x W - image pixel tensor
        faces (torch.LongTensor) 1 x Ntri x 3 - faces indices for mesh
            Must be copied to batch size.
    Output:
        texture B x N x 2 - texture colors unprojected on (assigned to) vertices
        vis_mask B x N - {0., 1.} mask of vertex visibility
    """
    ### map coordinates to NDC format
    img_size = img.size()[-2:]
    batch_size = verts3d.size(0)
    n_verts = verts3d.size(1)
    faces_batch = faces.repeat(batch_size, 1, 1).to(verts3d.device)
    verts3d = fit_vertices_to_orthographic(verts3d, img_size=img_size)
    meshes = convert_vertices_to_mesh(verts3d, faces_batch)

    ### compute visibility mask
    vis_mask = get_vertex_visibility_mask(meshes, cameras, img_size)
    vis_mask = vis_mask.view(verts3d.size(0), verts3d.size(1))

    ### extract colors
    verts_rgb = unproject_to_vertices(img, verts3d)
    return verts_rgb, vis_mask




