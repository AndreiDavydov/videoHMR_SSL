import numpy as np
import pytorch3d.renderer
import pytorch3d.structures
import torch


class SilhouetteRendererPytorch3d(torch.nn.Module):
    """
    This github issue raises the question on weird behavior of SoftSilhoetteShader:
    https://github.com/facebookresearch/pytorch3d/issues/470#issuecomment-921832189

    Its implementation indeed does not take into account z-coordinate when computes alpha blend.
    Hence, it brought up waves on the mesh silhouette, probably because it tried to aggregate colors
    from faces on different sides of the body.

    NOTE: Use SilhouetteRenderer instead!
    """

    def __init__(self, batch_size=1, img_size=(256, 256), device="cpu", cameras=None):
        super().__init__()

        self.img_size = img_size
        self.device = device

        ### cameras
        if cameras is None:
            #  by default, use orthographic camera
            cameras = get_default_cameras(device, mode="orthographic")
        self.cameras = cameras

        ### rasterizer
        self.sigma = 1e-5
        self.raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=img_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * self.sigma,
            faces_per_pixel=25,
        )
        self.rasterizer = pytorch3d.renderer.MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings,
        )
        self.shader = pytorch3d.renderer.SoftSilhouetteShader()

    def forward(self, mesh):
        """
        Args:
            mesh (pytorch3d.structures.Meshes): pytorch3d mesh
        """
        fragments = self.rasterizer(mesh)
        silhouette_out = self.shader(fragments, mesh)
        silhouettes = silhouette_out[..., 3]
        return silhouettes


class SilhouetteRenderer(torch.nn.Module):
    """
    This issue raises the question on weird behavior of SoftSilhoetteShader:
    https://github.com/facebookresearch/pytorch3d/issues/470#issuecomment-921832189

    Its implementation indeed does not take into account z-coordinate when computes alpha blend
    """

    def __init__(self, batch_size=1, img_size=(256, 256), device="cpu", cameras=None):
        super().__init__()

        self.img_size = img_size
        self.device = device

        ### cameras
        if cameras is None:
            #  by default, use orthographic camera
            cameras = get_default_cameras(device, mode="orthographic")
        self.cameras = cameras

        ### rasterizer
        self.raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.rasterizer = pytorch3d.renderer.MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings,
        )
        self.blend_params = pytorch3d.renderer.BlendParams(background_color=(0, 0, 0))

        self.materials = pytorch3d.renderer.Materials(
            device=device,
            specular_color=[[0.0, 0.0, 0.0]],
            diffuse_color=[[0.0, 0.0, 0.0]],
        )
        self.shader = pytorch3d.renderer.SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            blend_params=self.blend_params,
            materials=self.materials,
        )

    def forward(self, mesh):
        """
        Args:
            mesh (pytorch3d.structures.Meshes): pytorch3d mesh
        """
        ### recolor mesh vertices to 1.
        verts_rgb = torch.ones(mesh.verts_padded().shape, requires_grad=False).to(
            mesh.device
        )
        mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)

        fragments = self.rasterizer(mesh)
        silhouette_out = self.shader(fragments, mesh)
        silhouettes = silhouette_out[..., 0] * 2  # TODO why its values are 0 and 0.5?
        return silhouettes


class ColoredRenderer(torch.nn.Module):
    def __init__(
        self,
        batch_size=1,
        img_size=(256, 256),
        device="cpu",
        specular_color=False,
        diffuse_color=False,
        cameras=None,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device

        ### cameras
        if cameras is None:
            #  by default, use orthographic camera
            cameras = get_default_cameras(device, mode="orthographic")
        self.cameras = cameras

        ### rasterizer
        self.raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.rasterizer = pytorch3d.renderer.MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings,
        )
        self.lights = pytorch3d.renderer.PointLights(
            device=device, location=[[0.0, 0.0, 0.0]]
        )
        self.blend_params = pytorch3d.renderer.BlendParams(background_color=(0, 0, 0))

        self.specular_color = specular_color
        specular_color = [[1] * 3] if self.specular_color else [[0] * 3]
        self.diffuse_color = diffuse_color
        diffuse_color = [[1] * 3] if self.diffuse_color else [[0] * 3]
        self.materials = pytorch3d.renderer.Materials(
            device=device,
            specular_color=specular_color,
            diffuse_color=diffuse_color,
        )
        self.shader = pytorch3d.renderer.SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=self.lights,
            blend_params=self.blend_params,
            materials=self.materials,
        )

    def forward(self, mesh):
        """
        Args:
            mesh (pytorch3d.structures.Meshes): pytorch3d mesh
        """
        fragments = self.rasterizer(mesh)
        images_out = self.shader(fragments, mesh)
        images_out = images_out[..., :3]
        if not (self.specular_color or self.diffuse_color):
            # if specular_color and diffuse_color both False, max brightness is 0.5
            # TODO emprically found. there is probably an explanation
            images_out = images_out * 2
        return images_out


def fit_vertices_to_orthographic(vertices, img_size=(224, 224)):
    """
    A bypass for proper orthographic projection of mesh vertices.
    Args:
        vertices : batch_size x N_verts x 3
        img_size : tuple, (H,W)
    """
    out = torch.zeros_like(vertices)
    out[:, :, 0] = vertices[:, :, 0] - img_size[1] / 2
    out[:, :, 1] = vertices[:, :, 1] - img_size[0] / 2
    out[:, :, 2] = vertices[:, :, 2]

    out[:, :, 0:2] = -1 * out[:, :, 0:2]

    min_size = min(img_size)
    out = out / (min_size / 2)
    out[:, :, 2] = out[:, :, 2] + 10

    return out


def convert_vertices_to_mesh(vertices, faces, verts_rgb_colors=None):
    """
    vertices (torch.tensor) : B x N x 3
    faces (np.array) : B x Ntri x 3
    """
    device = vertices.device

    if verts_rgb_colors is None:
        verts_rgb_colors = 0.6 * torch.ones(vertices.size()).to(device)
        verts_rgb_colors[..., 0] += 0.4
    tex = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb_colors)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=tex,
    ).to(device)

    return mesh


def get_default_cameras(device, mode="orthographic"):
    if mode == "orthographic":
        cameras = pytorch3d.renderer.FoVOrthographicCameras(device=device, zfar=100)
    elif mode == "perspective":
        cam_distance = 2.4
        batch_size = 1
        R, T = pytorch3d.renderer.look_at_view_transform(cam_distance, 0, 0)
        R = R.repeat(batch_size, 1, 1)  # B x 3 x 3
        T = T.repeat(batch_size, 1)  # B x 3
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        raise NotImplementedError

    return cameras


def render_mesh_onto_image(img, mesh_img):  # , alpha=0.3):
    """
    Args:
        img1, img2 (torch.tensor): tensors of the same shape
        alpha (float): mixing coefficient
    """
    mesh_img_mask = mesh_img > 0
    blend_img = mesh_img_mask * mesh_img + (~mesh_img_mask) * img
    return blend_img


def unproject_to_faces(img, mesh, img_size):
    """
    Currently works with RGB images only.
    TODO Other modalities (e.g. Optical FLow) should be checked separately!
    """
    num_faces = mesh.faces_padded().shape[1]
    pix_to_face = pytorch3d.renderer.rasterize_meshes(
        mesh, img_size, faces_per_pixel=1
    )[0]
    best_face_for_pixel = pix_to_face[..., 0]
    textures = unproject_img_pixels_onto_mesh_faces(
        img, best_face_for_pixel, num_faces=num_faces, img_size=img_size
    )
    return textures


def unproject_img_pixels_onto_mesh_faces(img, best_face_for_pixel, num_faces, img_size):

    device = img.device
    batch_size = best_face_for_pixel.size(0)

    faces_rgbs = []

    for batch_idx in range(batch_size):
        rows, cols = torch.meshgrid(
            torch.arange(img_size[0]), torch.arange(img_size[1])
        )
        rows = rows.to(device)
        cols = cols.to(device)

        best_face_for_pixel_per_img = best_face_for_pixel[batch_idx]
        best_face_for_pixel_mask = best_face_for_pixel_per_img > -1
        best_face_for_pixel_masked = best_face_for_pixel_per_img[
            best_face_for_pixel_mask
        ]
        rows = rows[best_face_for_pixel_mask]
        cols = cols[best_face_for_pixel_mask]

        ### remove batch_idx from face ids:
        best_face_for_pixel_masked = best_face_for_pixel_masked - int(
            batch_idx * num_faces
        )
        faces_rgb = torch.zeros((num_faces, 3)).to(device)
        faces_rgb[best_face_for_pixel_masked] = img[batch_idx, rows, cols]

        faces_rgbs.append(faces_rgb)

    faces_rgbs = torch.stack(faces_rgbs).unsqueeze(2).unsqueeze(2)  # B x F x 1 x 1 x 3

    textures = pytorch3d.renderer.TexturesAtlas(faces_rgbs)
    return textures


def unproject_to_vertices(data, vertices):
    """
    Args:
        data : B x C x H x W - data to sample from.
            Using uint8 format is prohibited. Float works fine.
        vertices : B x N x 3 (in NDC format, ready to rasterize
            N == number of vertices
    """
    sample_at = -vertices.unsqueeze(2)[..., :2]  # B x N x 1 x 2
    data_sampled = torch.nn.functional.grid_sample(data, sample_at)  # B x C x N x 1
    textures = data_sampled[..., 0].permute(0, 2, 1)  # B x N x C
    return textures


def get_vertex_visibility_mask(meshes, cameras, img_size):
    ### taken from https://github.com/facebookresearch/pytorch3d/issues/126
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings
    )

    # Get the output from rasterization
    fragments = rasterizer(meshes)

    # pix_to_face is of shape (N, H, W, 1)
    pix_to_face = fragments.pix_to_face

    # (F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = meshes.faces_packed()
    # (V, 3) where V is the total number of verts across all the meshes in the batch
    packed_verts = meshes.verts_packed()
    device = packed_verts.device
    vertex_visibility_mask = torch.zeros(packed_verts.shape[0]).to(device)  # (V,)

    # Indices of unique visible faces
    visible_faces = pix_to_face.unique()  # (num_visible_faces )

    # Get Indices of unique visible verts using the vertex indices in the faces
    visible_verts_idx = packed_faces[visible_faces]  # (num_visible_faces,  3)
    unique_visible_verts_idx = torch.unique(visible_verts_idx)  # (num_visible_verts, )

    # Update visibility indicator to 1 for all visible vertices
    vertex_visibility_mask[unique_visible_verts_idx] = 1.0
    return vertex_visibility_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from iopath.common.file_io import PathManager
    # from iopath.fb.manifold import ManifoldPathHandler

    # pathmgr = PathManager()
    # pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

    ### test renderers
    test_renderers = True
    if test_renderers:

        sample_path = "manifold://xr_body/tree/personal/andreydavydov/renderer/sample_for_tests.pth"
        # d = torch.load(pathmgr.get_local_path(sample_path))
        d = torch.load(sample_path)

        device = "cuda:0"
        img = d["img"].clone().to(device)  # B x 3 x H x W
        vertices = d["bbox_verts"].clone().to(device)  # B x N x 3
        faces = d["faces"]  ### type: int64 (result of smpl_model.faces.astype(int))

        torch.set_grad_enabled(False)

        batch_size = img.size(0)
        img_size = img.shape[-2:]

        faces_batch = (
            torch.tensor(faces.copy()).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        )

        ### new version of "render onto image" - torch and batch-wise
        img = img.permute(0, 2, 3, 1) / 255  # B x H x W x 3

        colorbody_renderer = ColoredRenderer(
            img_size=img_size, device=device, specular_color=True, diffuse_color=True
        )
        silh_renderer = SilhouetteRenderer(img_size=img_size, device=device)
        silh_renderer_old = SilhouetteRendererPytorch3d(
            img_size=img_size, device=device
        )
        texel_renderer = ColoredRenderer(img_size=img_size, device=device)

        ### for orthographic projection, body should be moved a bit away from the camera
        vertices = fit_vertices_to_orthographic(vertices, img_size)
        meshes = convert_vertices_to_mesh(vertices, faces_batch)
        render_images = colorbody_renderer(meshes)
        silh_images = silh_renderer(meshes.clone())
        silh_images_old = silh_renderer_old(meshes.clone())

        ### unproject img as texture to faces
        meshes_img = meshes.clone()
        meshes_img.textures = unproject_to_faces(img, meshes_img, img_size)
        img_to_faces = texel_renderer(meshes_img)

        ### to verts
        verts_rgb = unproject_to_vertices(img.permute(0, 3, 1, 2), vertices)
        meshes = convert_vertices_to_mesh(
            vertices, faces_batch, verts_rgb_colors=verts_rgb
        )
        img_to_verts = texel_renderer(meshes)

        blend_images = render_mesh_onto_image(img, render_images)
        blend_images_silh = render_mesh_onto_image(img, silh_images[..., None])
        blend_images_silh_old = render_mesh_onto_image(img, silh_images_old[..., None])
        blend_images__to_faces = render_mesh_onto_image(img * 0.6, img_to_faces)
        blend_images__to_verts = render_mesh_onto_image(img * 0.6, img_to_verts)

        images_to_plot = 10
        images = [
            img,
            blend_images__to_faces,
            blend_images__to_verts,
            blend_images,
            blend_images_silh_old,
            blend_images_silh,
        ]
        fig, ax = plt.subplots(
            images_to_plot, 6, figsize=(len(images) * 4, images_to_plot * 4)
        )
        for i in range(images_to_plot):
            for j, images_ in enumerate(images):
                ax[i, j].imshow(images_[i].cpu())

        for axis in ax.flatten():
            axis.set_axis_off()
        fig.tight_layout()
        fig.subplots_adjust()
        fig.patch.set_facecolor("white")

        savepath = "/tmp/renderer_test.png"
        plt.savefig(savepath, bbox_inches="tight")
        print(f"Figure is saved to {savepath}!")

    ### test vertex visibility mask
    test_vertex_vis_mask = True
    if test_vertex_vis_mask:
        import math

        sample_path = "manifold://xr_body/tree/personal/andreydavydov/renderer/sample_for_tests.pth"
        # d = torch.load(pathmgr.get_local_path(sample_path))
        d = torch.load(sample_path)

        device = "cuda:0"
        img = d["img"].clone().to(device)[:1]  # B x 3 x H x W
        img_size = img.shape[-2:]
        img = img.permute(0, 2, 3, 1) / 255  # B x H x W x 3

        vertices = d["bbox_verts"][:1].clone().to(device)  # B x N x 3
        faces = d["faces"]  ### type: int64 (result of smpl_model.faces.astype(int))

        torch.set_grad_enabled(False)

        batch_size = img.size(0)

        ### get the mask
        vertices = d["bbox_verts"].clone().to(device)
        vertices = fit_vertices_to_orthographic(vertices, img_size)
        meshes = convert_vertices_to_mesh(vertices, faces_batch)
        colorbody_renderer = ColoredRenderer(img_size=img_size, device=device)

        visibility_mask = get_vertex_visibility_mask(
            meshes, colorbody_renderer.cameras, img_size
        )
        verts_rgb_colors = torch.ones_like(vertices).view(-1, 3) * 0.3
        verts_rgb_colors[visibility_mask > 0.5] = 1.0
        verts_rgb_colors = verts_rgb_colors.view(vertices.size())

        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        rot_images = []
        for angle in angles:

            c = math.cos(angle / 360 * 2 * math.pi)
            s = math.sin(angle / 360 * 2 * math.pi)

            vertices_ = vertices.clone()
            verts_x = vertices_[:, :, 0] * c - (vertices_[:, :, 2] - 10) * s
            verts_z = vertices_[:, :, 0] * s + (vertices_[:, :, 2] - 10) * c
            vertices_[:, :, 0] = verts_x
            vertices_[:, :, 2] = verts_z + 10

            meshes = convert_vertices_to_mesh(vertices_, faces_batch)
            textures = pytorch3d.renderer.TexturesVertex(
                verts_features=verts_rgb_colors
            )
            meshes.textures = textures

            render_images = colorbody_renderer(meshes)
            blend_images = render_mesh_onto_image(img, render_images)
            rot_images.append(blend_images[0].cpu()[:, 30:-30])

        rot_images = torch.cat(rot_images, dim=1)
        fig, ax = plt.subplots(1, 1, figsize=(len(angles) * 4, 4))
        ax.imshow(rot_images)
        ax.set_axis_off()

        fig.patch.set_facecolor("white")

        savepath = "/tmp/vis_mask_test.png"
        plt.savefig(savepath, bbox_inches="tight")
        print(f"Figure is saved to {savepath}!")
