import matplotlib.pyplot as plt

from src.datasets.datasets_common import UNNORMALIZE
from src.utils.vis_utils import render_mesh_onto_image


def plot_batch_with_mesh(
    img,
    verts_bbox,
    faces,
    n_imgs=5,
    figsize=(15, 15),
    unnorm_img=True,
):
    assert (
        img.size(0) >= n_imgs**2
    ), f"cannot draw {n_imgs}x{n_imgs} figure given {img.size(0)} images in the batch"

    if unnorm_img:
        img = UNNORMALIZE(img)

    fig, ax = plt.subplots(n_imgs, n_imgs, figsize=figsize)
    for i in range(n_imgs):
        for j in range(n_imgs):
            k = i * n_imgs + j
            img_ = img[k].permute(1, 2, 0).numpy()
            verts_bbox_ = verts_bbox[k].numpy()
            blend_img = render_mesh_onto_image(img_, verts_bbox_, faces)
            ax[i, j].imshow(blend_img)
            ax[i, j].set_axis_off()

    plt.subplots_adjust()
    plt.tight_layout()

    return fig
