import torch
# from body_tracking.smplx.lbs import batch_rodrigues
from smplx.lbs import batch_rodrigues


class SmplPoseLoss(torch.nn.Module):
    def __init__(self, num_joints=24):
        super(SmplPoseLoss, self).__init__()
        self.num_joints = num_joints

    def forward(self, pred_rotmat, gt_aa):
        """
        pred_rotmat: Predicted poses in rot mat (B x 24 x 3 x 3).
        gt_aa: GT poses in axis-angle (B x 72).
        """
        batch_size = pred_rotmat.shape[0]
        gt_poses = batch_rodrigues(gt_aa.view(-1, 3)).view(
            batch_size, self.num_joints * 9
        )
        pred_poses = pred_rotmat.view(batch_size, self.num_joints * 9)
        loss = ((pred_poses - gt_poses) ** 2).mean()  # l2 loss
        return loss


class SmplShapeLoss(torch.nn.Module):
    def __init__(self):
        super(SmplShapeLoss, self).__init__()

    def forward(self, pred_shape, gt_shape):
        return ((pred_shape - gt_shape) ** 2).mean()  # l2 loss


class SmplShapePriorLoss(torch.nn.Module):
    def __init__(self):
        super(SmplShapePriorLoss, self).__init__()

    def forward(self, pred_shape):
        return (pred_shape**2).mean()  # l2 norm


class Kpts2dLoss(torch.nn.Module):
    def __init__(self):
        super(Kpts2dLoss, self).__init__()

    def forward(self, pred_2d, gt_2d):
        vis = (gt_2d[..., 2:] > 0).float()  # B x J x 1
        loss = vis * ((pred_2d - gt_2d[..., :2]) ** 2)
        loss = loss.sum() / vis.sum()
        return loss


class Kpts3dLoss(torch.nn.Module):
    def __init__(self):
        super(Kpts3dLoss, self).__init__()

    def forward(self, pred_3d, gt_3d):
        pred_3d = align_by_pelvis(pred_3d)
        gt_3d = align_by_pelvis(gt_3d)
        loss = ((pred_3d - gt_3d) ** 2).mean()
        return loss


class CameraScaleRegLoss(torch.nn.Module):
    def __init__(self):
        super(CameraScaleRegLoss, self).__init__()

    def forward(self, camera_scale):
        return ((torch.exp(-camera_scale * 10)) ** 2).mean()


def align_by_pelvis(joints):
    """
    Pelvis is midpoint of hips (indices 2 and 3).
    Subtract pelvis from the skeleton.

    Args:
        joints (tensor): 3D Joints (B x J x 3).
    """
    right_hip_id = 2
    left_hip_id = 3
    pelvis = (joints[:, right_hip_id] + joints[:, left_hip_id]) / 2
    return joints - torch.unsqueeze(pelvis, dim=1)
