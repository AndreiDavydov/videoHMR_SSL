import pytorch3d.renderer
import torch
from src.functional.renderer import (
    convert_vertices_to_mesh,
    fit_vertices_to_orthographic,
    get_vertex_visibility_mask,
    unproject_to_vertices,
)


def unproject_optical_flows_to_vertices(verts3d, of, faces, cameras):
    """
    Args:
        verts3d (torch.tensor) B x N x 3 - mesh vertices, start/end together.
            X,Y coordinates are in pixels.
        of (torch.tensor) B x 2 x H x W - optical flow
            (e.g., from the pretrained optical flow predictor)
        faces (torch.LongTensor) 1 x Ntri x 3 - faces indices for mesh
            Must be copied to batch size.
    Output:
        unproj_flow2d B x N x 2 - 2d flow unprojected on (assigned to) vertices
        vis_mask B x N - {0., 1.} mask of vertex visibility
    """
    ### map coordinates to NDC format
    img_size = of.size()[-2:]
    batch_size = verts3d.size(0)
    n_verts = verts3d.size(1)
    faces_batch = faces.repeat(batch_size, 1, 1).to(verts3d.device)
    verts3d = fit_vertices_to_orthographic(verts3d, img_size=img_size)
    meshes = convert_vertices_to_mesh(verts3d, faces_batch)

    ### compute visibility mask
    vis_mask = get_vertex_visibility_mask(meshes, cameras, img_size)
    vis_mask = vis_mask.view(verts3d.size(0), verts3d.size(1))

    ### unproject optical flow to vertices
    unproj_flow2d = unproject_to_vertices(of, verts3d)
    return unproj_flow2d, vis_mask


def get_of(model, frames_start, frames_end, device):
    """
    Compute optical flow using off-the-shelf model.
    
    NOTE: frames_start and frames_end should be unnormalized! 
    """
    img1 = frames_start.to(device)
    img2 = frames_end.to(device)

    with torch.no_grad():
        # compute optical flow
        _, opt_flow = model(img1, img2, iters=20, test_mode=True)
    return opt_flow
    

### OLD
def unproject_optical_flows_to_vertices_Batch_sequence(
    verts3d, of_forward, of_backward, faces, cameras
):
    """
    Args:
        verts3d (torch.tensor) B x N x 3 - mesh vertices.
            X,Y coordinates are in pixels.
        of_forward, of_backward (torch.tensor) B-1 x 2 x H x W - optical flows
            (e.g., from the pretrained optical flow predictor)
        faces (torch.LongTensor) 1 x Ntri x 3 - faces indices for mesh
            Must be copied to batch size.
    Output:
        unproj_flow2d_forward, unproj_flow2d_backward B-1 x N x 2
            - 2d flows unprojected on vertices
        vis_mask B-1 x N - {0., 1.} mask of vertex visibility
    """
    ### map coordinates to NDC format
    img_size = of_forward.size()[-2:]
    batch_size = verts3d.size(0)
    faces_batch = faces.repeat(batch_size, 1, 1).to(verts3d.device)
    verts3d = fit_vertices_to_orthographic(verts3d, img_size=img_size)
    meshes = convert_vertices_to_mesh(verts3d, faces_batch)

    ### compute visibility mask
    vis_mask = get_vertex_visibility_mask(meshes, cameras, img_size)
    vis_mask = vis_mask.view(verts3d.size(0), verts3d.size(1))
    #  mask should be the intersection of neighboring masks
    vis_mask = vis_mask[:-1] * vis_mask[1:]  # B-1 x N  (B - # of frames)

    verts3d_start = verts3d[:-1]
    verts3d_end = verts3d[1:]

    ### unproject optical flow to vertices
    #  NOTE: flow values are in pixels!
    unproj_flow2d_forward = unproject_to_vertices(of_forward, verts3d_start)
    unproj_flow2d_backward = unproject_to_vertices(of_backward, verts3d_end)
    return unproj_flow2d_forward, unproj_flow2d_backward, vis_mask


####################################
### 3d flow loss routine
####################################


def unproject_data_to_vertices(verts3d, data, faces, cameras):
    """
    Args:
        verts3d (torch.tensor) B x N x 3 - mesh vertices.
            X,Y coordinates are in pixels.
        data (torch.tensor) B x C x H x W - data (can be OF)
            (e.g., from the pretrained optical flow predictor)
        faces (torch.LongTensor) 1 x Ntri x 3 - faces indices for mesh
            Must be copied to batch size.
    Output:
        unproj_data B x N x C
        vis_mask B x N - {0., 1.} mask of vertex visibility
    """
    ### map coordinates to NDC format
    img_size = data.size()[-2:]
    batch_size = verts3d.size(0)
    faces_batch = faces.repeat(batch_size, 1, 1).to(verts3d.device)
    verts3d = fit_vertices_to_orthographic(verts3d, img_size=img_size)
    meshes = convert_vertices_to_mesh(verts3d, faces_batch)

    ### compute visibility mask
    vis_mask = get_vertex_visibility_mask(meshes, cameras, img_size)
    vis_mask = vis_mask.view(verts3d.size(0), verts3d.size(1))

    ### unproject optical flow to vertices
    #  NOTE: flow values are in pixels!
    unproj_flow2d_forward = unproject_to_vertices(data, verts3d)
    return unproj_flow2d_forward, vis_mask


def get_corners_for_points(points):
    """
    For every point:
        1. find corresponding pixels quadruplet
        2. compute dx,dy coordinates in the corner pixels square (>=0, <=1)
    Args:
        points : B x N x 2 - float coordinates to be reprojected
    """
    ### find 4 pixel coordinates that are keeping "points"
    # Example: p (from points) = (0.1, 1.2) = (x, y)
    # -> tl = (0, 0)
    # -> tr = (1, 0)
    # -> bl = (0, 1)
    # -> br = (1, 1)

    ### keep coordinates inside 1 px cell
    tl_x = torch.floor(points[..., 0])
    tl_y = torch.floor(points[..., 1])

    ### compute cell corners
    br_x = torch.ceil(points[..., 0])
    br_y = torch.ceil(points[..., 1])
    tl = torch.stack((tl_x, tl_y), dim=-1)
    br = torch.stack((br_x, br_y), dim=-1)
    tr = torch.stack((br_x, tl_y), dim=-1)
    bl = torch.stack((tl_x, br_y), dim=-1)

    ### tl - tr - bl - br order
    px_quads = torch.stack((tl, tr, bl, br), dim=-1).long()

    ### reshape to B x N x 4 x 2 for masking
    px_quads = px_quads.permute(0, 1, 3, 2).contiguous()

    ### compute the coordinates of points
    dx = points[..., 0] - tl_x
    dy = points[..., 1] - tl_y
    return px_quads, dx, dy


def unproject_corners_on_mesh(corners, verts, faces, cameras, img_size=(224, 224)):
    """
    For each corner pixel find:
        the closest face (if any);
        corresponding barycentric coordinates;
        compute visibility mask.

        If face is not visible, then all 4 corners should not be considered.

    Args:
        corners : B x N x 4 x 2 (torch.int64) - quadruplet of pixel indices
        verts : B x N x 3 - underlying mesh vertices onto which to project the corners

    """
    ### rasterize mesh
    faces_batch = faces.repeat(verts.size(0), 1, 1).to(verts.device)
    verts3d_orth = fit_vertices_to_orthographic(verts, img_size=img_size)
    meshes = convert_vertices_to_mesh(verts3d_orth, faces_batch)
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings
    )
    fragments = rasterizer(meshes)
    pix_to_face = fragments.pix_to_face
    bary_coords = fragments.bary_coords

    ### for each corner find faces, bary_coords, vis_mask
    # define N' = B*N
    Nverts = verts.size(1)
    vis_mask = []  ## mask N' x 4
    bary_coords_corners = []  ## N' x 4 x 3
    verts_triplet_indices = []
    batch_ids = torch.arange(corners.size(0)).view(-1, 1).repeat(1, Nverts).view(-1)
    for corner_idx in range(4):
        x_ids = corners[..., corner_idx, 0].view(-1)
        y_ids = corners[..., corner_idx, 1].view(-1)

        faces_ = pix_to_face[batch_ids, x_ids, y_ids, 0]
        verts_triplet_indices_ = faces_batch.view(-1, 3)[faces_]
        vis_mask.append((faces_ >= 0).type(torch.bool))
        bary_coords_ = bary_coords[batch_ids, x_ids, y_ids, 0, :]
        bary_coords_corners.append(bary_coords_)
        verts_triplet_indices.append(verts_triplet_indices_)

    #  if any of of four corners is invisible, then no processing
    vis_mask = torch.stack(vis_mask, dim=-1).prod(dim=-1)  # N'
    bary_coords_corners = torch.stack(bary_coords_corners, dim=-1)
    bary_coords_corners = bary_coords_corners.permute(0, 2, 1)  # N' x 4 x 3
    verts_triplet_indices = torch.stack(verts_triplet_indices, dim=-1)
    verts_triplet_indices = verts_triplet_indices.permute(0, 2, 1)  # N' x 4 x 3

    return bary_coords_corners, verts_triplet_indices, vis_mask
