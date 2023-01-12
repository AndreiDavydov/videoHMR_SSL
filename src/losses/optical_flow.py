import torch


class Flow2DLoss(torch.nn.Module):
    def __init__(self):
        super(Flow2DLoss, self).__init__()

    def forward(self, verts2d_flow_pred, optical_flow_2d_unproj, vis_mask):
        """
        Both tensors verts2d_flow_pred and optical_flow_2d_unproj
            have size B x N x 2.
        vis_mask : torch.tensor B x N with 0s and 1s.
        """
        err = (verts2d_flow_pred - optical_flow_2d_unproj) ** 2
        err = err.sum(dim=-1)  # B x N
        err = err * vis_mask
        return err.sum() / vis_mask.sum()


# class Flow3DLoss(torch.nn.Module):
#     def __init__(self):
#         super(Flow3DLoss, self).__init__()

#     def forward(self, weights, mask_map_to_same_indices):
#         same_w = weights[mask_map_to_same_indices]
#         diff_w = weights[~mask_map_to_same_indices]

#         ### "same" terms should be pulled to 1
#         ### "diff" terms should be pulled to 0 * geodesic
#         print(weights.shape, mask_map_to_same_indices.shape)
#         print(same_w.shape, diff_w.shape)
#         loss_same = ((same_w - 1) ** 2).sum()
#         loss_diff = (diff_w**2).sum()

#         loss = loss_same + loss_diff
#         loss = loss / weights.numel()

#         ### TODO multiply loss_diff by geodesic distances!

#         return loss
