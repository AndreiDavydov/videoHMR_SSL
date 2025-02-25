from time import time

import torch

import src 
from src.datasets.datasets_common import UNNORMALIZE
from src.procedures.procedures_common import status_msg

from src.functional.hmr import hmr_inference
from src.functional.texturepose import unproject_texture_to_vertices
from src.procedures.procedures.finetune_hmr_ssl_w_single_frame_gt import setup, train_frame
from src.procedures.procedures.finetune_hmr_ssl_w_single_frame_gt_seqlen2 import hmr_inference
from src.procedures.procedures.finetune_hmr_ssl_w_single_frame_gt_seqlen2 import valid


def train_texturepose(sample, trainer):
    device = trainer.device0

    ### turn off BN 
    trainer.models.hmrnet.eval()

    # take one video sequence: B x seqlen==2 x 3 x 224 x 224
    img = sample["video"]
    batch_size = img.size(0)
    img_size = img.size(-1)

    ### inference
    img = img.to(device,non_blocking=True)
    out = hmr_inference(img.flatten(start_dim=0, end_dim=1), trainer.models.hmrnet, trainer.smpl_model_49)
    verts3d = out['verts3d']

    ### compute textures
    img_rgb = UNNORMALIZE(img.flatten(start_dim=0, end_dim=1)) / 255
    verts_rgb, vis_mask = unproject_texture_to_vertices(
        verts3d, img_rgb, trainer.smpl_model_faces, trainer.cameras)
    verts_rgb = verts_rgb.view(batch_size, 2, verts_rgb.size(-2), 3)  # B x 2 x N x 3
    
    vis_mask = vis_mask.view(batch_size, 2, verts_rgb.size(-2))  # B x 2 x N
    vis_mask = vis_mask[:,0] * vis_mask[:,1]  # B x N
    vis_mask = vis_mask.unsqueeze(-1)  # B x N x 1

    ### simple l2 loss
    loss = ( ( verts_rgb[:,0] - verts_rgb[:,1] ) **2 ) * vis_mask
    loss = loss.sum() / vis_mask.sum()

    loss = loss * trainer.losses_weights.texture_cons
    trainer.meters.train.texture_cons.update_raw(loss.item())

    return loss


def train(trainer):

    absolute_start = time()
    
    dl_len = len(trainer.dataload.multidl)
    for batch_idx, (sample_frame, sample_video) in enumerate(trainer.dataload.multidl, start=1):
        
        ### inference for training with GT per frame
        train_frame_loss = train_frame(sample_frame, trainer)

        ### inference for training with TexturePose guidance 
        train_texturepose_loss = train_texturepose(sample_video, trainer)

        ### weight frame / video terms
        loss = trainer.cfg.W_FRAME * train_frame_loss + trainer.cfg.W_TP * train_texturepose_loss

        ### do backprop
        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()

        trainer.meters.train.full.update(loss.item(), n=1)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.full, total_time)
        if batch_idx != dl_len: trainer.meters.train.full.epochends()
    