import matplotlib.pyplot as plt
import numpy as np
import torch

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler

# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)


def plot_mpjpes(ckpt_names):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    for name, full_name in ckpt_names.items():
        ckpt_dir = f"manifold://xr_body/tree/personal/andreydavydov/my_exps/{full_name}"

        pa_mpjpe_vs_gt_valid = torch.load(
            # pathmgr.get_local_path(
            #     f"{ckpt_dir}/metrics/pa_mpjpe_vs_gt_valid.pth", force=True
            # )
                f"{ckpt_dir}/metrics/pa_mpjpe_vs_gt_valid.pth"
        )["prev_vals"]
        ax[0].plot(pa_mpjpe_vs_gt_valid, label=name)
        pa_mpjpe_vs_0_valid = torch.load(
            # pathmgr.get_local_path(
            #     f"{ckpt_dir}/metrics/pa_mpjpe_vs_0_valid.pth", force=True
            # )
            f"{ckpt_dir}/metrics/pa_mpjpe_vs_0_valid.pth"
        )["prev_vals"]
        ax[1].plot(pa_mpjpe_vs_0_valid, label=name)
        for axis in ax:
            axis.legend(fontsize=15)
        fig.patch.set_facecolor("white")

    ax[0].set_title("PA-MPJPE Opt vs GT, mm", fontsize=20)
    ax[1].set_title("PA-MPJPE Opt vs Init, mm", fontsize=20)

    return fig


def plot_accel(ckpt_names, force=True):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for name, full_name in ckpt_names.items():
        ckpt_dir = f"manifold://xr_body/tree/personal/andreydavydov/my_exps/{full_name}"
        accel_err_vs_gt_valid = torch.load(
            # pathmgr.get_local_path(
            #     f"{ckpt_dir}/metrics/accel_err_vs_gt_valid.pth", force=force
            # )
            f"{ckpt_dir}/metrics/accel_err_vs_gt_valid.pth"
        )["prev_vals"]
        ax.plot(accel_err_vs_gt_valid, label=f"vs GT {name}")

    fig.legend(fontsize=15)
    fig.patch.set_facecolor("white")
    return fig


def plot_mpjpes_and_accel(ckpt_names, force=True):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    for name, full_name in ckpt_names.items():
        ckpt_dir = f"manifold://xr_body/tree/personal/andreydavydov/my_exps/{full_name}"

        pa_mpjpe_vs_gt_valid = torch.load(
            # pathmgr.get_local_path(
            #     f"{ckpt_dir}/metrics/pa_mpjpe_vs_gt_valid.pth", force=force
            # )
            f"{ckpt_dir}/metrics/pa_mpjpe_vs_gt_valid.pth"
        )["prev_vals"]
        ax[0].plot(pa_mpjpe_vs_gt_valid, label=name)
        ax[0].set_ylabel("pa-mpjpe")

        accel_err_vs_gt_valid = torch.load(
            # pathmgr.get_local_path(
            #     f"{ckpt_dir}/metrics/accel_err_vs_gt_valid.pth", force=force
            # )
            f"{ckpt_dir}/metrics/accel_err_vs_gt_valid.pth"
        )["prev_vals"]
        ax[1].plot(accel_err_vs_gt_valid, label=f"vs GT {name}")
        ax[1].set_ylabel("acceleration")

        for axis in ax:
            axis.set_xlabel("epochs")
            # axis.legend(fontsize=15)
        fig.patch.set_facecolor("white")

    ax[0].set_title("PA-MPJPE Opt vs GT, mm", fontsize=20)
    ax[1].set_title(r"Accel Err Opt vs GT, $\frac{mm}{s^2}$", fontsize=20)

    return fig


def plot_shape_pose_norm(ckpt_names):

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    init_plotted = False
    for name, full_name in ckpt_names.items():
        ckpt_dir = f"manifold://xr_body/tree/personal/andreydavydov/my_exps/{full_name}"
        ckpt = torch.load(
            # pathmgr.get_local_path(f"{ckpt_dir}/ckpt.pth", force=True),
            f"{ckpt_dir}/ckpt.pth",
            map_location="cpu",
        )

        if not init_plotted:
            ax[0].plot(
                ckpt["seqOpt_state_dict"]["shape_init"]
                .flatten(start_dim=1)
                .norm(dim=-1),
                c="green",
                label="Init",
                linewidth=5,
            )
            ax[1].plot(
                ckpt["seqOpt_state_dict"]["pose_init"]
                .flatten(start_dim=1)
                .norm(dim=-1),
                c="green",
                label="Init",
                linewidth=5,
            )
            init_plotted = True

        ax[0].plot(
            ckpt["seqOpt_state_dict"]["shape"].flatten(start_dim=1).norm(dim=-1),
            label=f"{name}",
            linewidth=3,
        )
        ax[0].set_title("SMPL shape (norm)", fontsize=25)

        ax[1].plot(
            ckpt["seqOpt_state_dict"]["pose"].flatten(start_dim=1).norm(dim=-1),
            label=f"{name}",
            linewidth=3,
        )
        ax[1].set_title("SMPL pose (aa, norm)", fontsize=25)

        for axis in ax:
            axis.legend(fontsize=15, loc="lower right")

    fig.patch.set_facecolor("white")

    return fig


def plot_heatmap(
    grid_matrix, y_labels, x_labels, y_axis_name="", x_axis_name="", title=""
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(grid_matrix, interpolation="nearest", cmap="rainbow")

    ax.set_yticks(np.arange(grid_matrix.shape[0]))
    ax.set_yticklabels(y_labels)
    ax.set(ylim=(min(ax.get_yticks()), max(ax.get_yticks())))
    ax.set_ylabel(y_axis_name)

    ax.set_xticks(np.arange(grid_matrix.shape[1]))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set(xlim=(min(ax.get_xticks()), max(ax.get_xticks())))
    ax.set_xlabel(x_axis_name)

    _ = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.patch.set_facecolor("white")
    ax.set_title(title)
    ax.grid(False)

    return fig, ax
