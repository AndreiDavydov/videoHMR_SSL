import torch


class Flow2DLoss(torch.nn.Module):
    """
    Two important parts: thresholding and normalization.

    Thresholding:
    Some of the pairs of frames are not valid 
    (due to severe occlusions or wrong frame cuts).
    Such samples can be discarded via thresholding of OF norm. 
    For every sample (pair of frames) in the batch one can compute OF norms. 
    The maximum value across all vertices seems to be a good thresholding.
    threshold by max: 50
    threshold by mean: 30 (tested on 3DPW-test samples)

    Normalization:
    Different pairs of frames from one or from different datasets might have
    very different motions (both body flow and OF). 
    Hence, it is important to equalize them for stable training.
    We propose to normalize by OF norm means (across all visible vertices).

    """
    def __init__(self, normalize_OF=False, do_thresholding=True, maxOF_threshold=50, reduction="mean"):
        super(Flow2DLoss, self).__init__()
        self.normalize_OF = normalize_OF
        self.do_thresholding = do_thresholding
        self.maxOF_threshold = maxOF_threshold
        self.reduction = reduction


    def forward(self, verts2d_flow_pred, optical_flow_2d_unproj, vis_mask):
        """
        Both tensors verts2d_flow_pred and optical_flow_2d_unproj
            have size B x N x 2.
        vis_mask : torch.tensor B x N with 0s and 1s.
        """
        ### it might happen that vis_mask is empty
        if vis_mask.sum() == 0:
            print("vis_mask is zero!")
            return verts2d_flow_pred[0,0,0] * 0.

        of_norm = optical_flow_2d_unproj.norm(dim=-1)  # B x N
        of_norm = of_norm * vis_mask  # B x N

        ### THRESHOLDING: mask bad OF values (might be due to wrong pairs of frames in a dataset)
        if self.do_thresholding:
            of_norm_max = of_norm.max(dim=-1).values  # B
            of_mask = torch.arange(of_norm_max.size(0)).to(of_norm_max.device)
            of_mask = of_mask[of_norm_max < self.maxOF_threshold]

            of_norm = of_norm[of_mask]
            verts2d_flow_pred = verts2d_flow_pred[of_mask]
            optical_flow_2d_unproj = optical_flow_2d_unproj[of_mask]
            vis_mask = vis_mask[of_mask]
        
        err = (verts2d_flow_pred - optical_flow_2d_unproj) ** 2  # B x N
        err = err.sum(dim=-1)  # B x N
        err = err * vis_mask  # B x N
        err = err.sum(dim=-1) / vis_mask.sum(dim=-1) # B
        
        ### NORMALIZATION: compute mean across all visible vertices per every sample
        if self.normalize_OF:
            of_norm_mean = of_norm.sum(dim=-1) / vis_mask.sum(dim=-1) # B
            s = of_norm_mean ** 2 # as error is ^2
            err = err / s.detach()  # B
        
        if sum(torch.isnan(err)) > 0:
            return verts2d_flow_pred[0,0,0] * 0.

        if self.reduction is None:
            return err
        elif self.reduction == "mean":
            return err.mean()




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
