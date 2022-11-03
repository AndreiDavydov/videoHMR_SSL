import numpy as np
import torch
from src.losses.train_hmr import align_by_pelvis


# taken from https://github.com/mkocabas/VIBE/blob/master/lib/utils/eval_utils.py
# function batch_compute_similarity_transform_torch
def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


class MPJPE(torch.nn.Module):
    def __init__(self, num_joints=14):
        super(MPJPE, self).__init__()
        self.num_joints = num_joints

    def forward(self, pred_kpts_3d, gt_kpts_3d):
        """
        pred_kpts_3d (tensor, B x J x 3): Predicted 3D poses
        gt_kpts_3d (tensor, B x J x 3): GT 3D poses

        NOTE : works properly for "h36m" joints system! (due to pelvis alignment)
        """
        ### align
        pred_kpts_3d = align_by_pelvis(pred_kpts_3d)
        gt_kpts_3d = align_by_pelvis(gt_kpts_3d)

        ### sqrt( x^2 + y^2 + z^2 )
        loss = torch.sqrt(((pred_kpts_3d - gt_kpts_3d) ** 2).sum(dim=-1))  # B x J
        loss = loss.mean(dim=-1)  # B
        return loss


class PAMPJPE(torch.nn.Module):
    def __init__(self, num_joints=14):
        super(PAMPJPE, self).__init__()
        self.num_joints = num_joints

    def forward(self, pred_kpts_3d, gt_kpts_3d):
        """
        pred_kpts_3d (tensor, B x J x 3): Predicted 3D poses
        gt_kpts_3d (tensor, B x J x 3): GT 3D poses

        NOTE : works properly for "h36m" joints system! (due to pelvis alignment)
        """
        ### align
        pred_kpts_3d = align_by_pelvis(pred_kpts_3d)
        gt_kpts_3d = align_by_pelvis(gt_kpts_3d)

        ### PA transform
        pred_kpts_3d = compute_similarity_transform(pred_kpts_3d, gt_kpts_3d)

        ### sqrt( x^2 + y^2 + z^2 )
        loss = torch.sqrt(((pred_kpts_3d - gt_kpts_3d) ** 2).sum(dim=-1))  # B x J
        loss = loss.mean(dim=-1)  # B
        return loss


class MPVPE(torch.nn.Module):
    def __init__(self):
        super(MPVPE, self).__init__()

    def forward(self, pred_verts, gt_verts):
        """
        pred_verts (tensor, B x Nverts x 3): Predicted mesh vertices
        gt_verts (tensor, B x Nverts x 3): GT mesh vetices
        """
        ### sqrt( x^2 + y^2 + z^2 )
        loss = torch.sqrt(((pred_verts - gt_verts) ** 2).sum(dim=-1))  # B x Nverts
        loss = loss.mean(dim=-1)  # B
        return loss


class PAMPVPE(torch.nn.Module):
    def __init__(self, num_verts=6890):
        super(PAMPVPE, self).__init__()
        self.num_verts = num_verts

    def forward(self, pred_verts_3d, gt_verts_3d):
        """
        pred_verts_3d (tensor, B x N x 3): Predicted mesh vertices
        gt_verts_3d (tensor, B x N x 3): GT mesh vertices
        """

        ### PA transform
        pred_verts_3d = compute_similarity_transform(pred_verts_3d, gt_verts_3d)

        ### sqrt( x^2 + y^2 + z^2 )
        loss = torch.sqrt(((pred_verts_3d - gt_verts_3d) ** 2).sum(dim=-1))  # B x N
        loss = loss.mean(dim=-1)  # B
        return loss


class Acceleration(torch.nn.Module):
    """Compute acceleration of 3D joints."""

    def __init__(self):
        super(Acceleration, self).__init__()

    def forward(self, joints):
        """
        joints (tensor, B x J x 3): 3D keypoints
        """
        velocities = joints[1:] - joints[:-1]
        acceleration = velocities[1:] - velocities[:-1]
        acceleration_normed = torch.linalg.norm(acceleration, dim=2)
        return acceleration_normed.mean(dim=1)  # B


class AccelerationError(torch.nn.Module):
    """Compute difference between accelerations
    of pred and gt 3D keypoint sequences."""

    def __init__(self):
        super(AccelerationError, self).__init__()

    def forward(self, joints_gt, joints_pred, vis=None):
        # taken from https://github.com/mkocabas/VIBE/blob/master/lib/utils/eval_utils.py
        """
        Computes acceleration error:
            1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (B x J x 3).
            joints_pred (B x J x 3).
            vis (B).
        Returns:
            error_accel (B-2).
        """

        ### align
        joints_pred = align_by_pelvis(joints_pred)
        joints_gt = align_by_pelvis(joints_gt)

        # TODO (if necessary) rewrite to pytorch
        joints_gt = joints_gt.cpu().numpy()
        joints_pred = joints_pred.cpu().numpy()

        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

        normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        else:
            invis = np.logical_not(vis)
            invis1 = np.roll(invis, -1)
            invis2 = np.roll(invis, -2)
            new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
            new_vis = np.logical_not(new_invis)

        err = np.mean(normed[new_vis], axis=1)
        err = torch.from_numpy(err)  # B
        return err


class Keypoint2DPixelAlignment(torch.nn.Module):
    def __init__(self):
        super(Keypoint2DPixelAlignment, self).__init__()
        self.unnorm = True

    def forward(self, pred_2d, gt_2d):

        vis = (gt_2d[..., 2] > 0).float()  # B x J
        gt_2d = gt_2d[..., :2]  # B x J x 2
        if self.unnorm:
            gt_2d = (gt_2d + 1) * 224 / 2
            pred_2d = (pred_2d + 1) * 224 / 2

        # sum over x,y
        loss = torch.sqrt(((pred_2d - gt_2d) ** 2).sum(dim=-1))
        loss = vis * loss
        loss = loss.sum(dim=-1) / vis.sum(dim=-1)
        return loss  # B
