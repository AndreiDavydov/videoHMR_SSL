import torch.nn as nn
import torch

from src.functional.geometry import rot6d_to_rotmat


def forward_with_context(
        x,  # input image
        gammas,  # context gammas
        betas,  # context betas
        hmrnet, # network
        n_iter=3,
        return_feature=False,
    ):
        batch_size = x.shape[0]

        init_pose = hmrnet.init_pose.expand(batch_size, -1)
        init_shape = hmrnet.init_shape.expand(batch_size, -1)
        init_cam = hmrnet.init_cam.expand(batch_size, -1)

        x = hmrnet.conv1(x)
        x = hmrnet.bn1(x)
        x = hmrnet.relu(x)
        x = hmrnet.maxpool(x)

        x1 = hmrnet.layer1(x)
        x2 = hmrnet.layer2(x1)
        x3 = hmrnet.layer3(x2)
        x4 = hmrnet.layer4(x3)

        xf = hmrnet.avgpool(x4)
        xf = xf.view(xf.size(0), -1)  # [N, 2048]
        
        ### Here we inject the context gammas and betas
        xf = xf * gammas + betas

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for _ in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = hmrnet.fc1(xc)
            xc = hmrnet.drop1(xc)
            xc = hmrnet.fc2(xc)
            xc = hmrnet.drop2(xc)
            pred_pose = hmrnet.decpose(xc) + pred_pose
            pred_shape = hmrnet.decshape(xc) + pred_shape
            pred_cam = hmrnet.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        if not return_feature:
            return pred_rotmat, pred_shape, pred_cam
        else:
            return pred_rotmat, pred_shape, pred_cam, xf


class FilmNet(torch.nn.Module):
    def __init__(self, ctx_len=1, in_features=85, hs1=16, hs2=128, output_size=2048):
        super(FilmNet, self).__init__()
        self.ctx_len = ctx_len
        self.conv1x1 = nn.Conv1d(self.ctx_len, hs1, kernel_size=1, bias=True)
        self.downsample = nn.Linear(in_features*hs1, hs2, bias=True)

        self.layer_gammas = nn.Linear(hs2, output_size, bias=False)
        self.layer_betas = nn.Linear(hs2, output_size, bias=False)
        
        self.nonlin = nn.LeakyReLU(0.2, inplace=True)
        self.output_size = output_size

        for p in self.parameters():
            torch.nn.init.normal_(p, mean=0., std=0.01)

    def forward(self, x):
        '''
        input is of size B x ctx_len x in_features
        '''
        x = self.nonlin(self.conv1x1(x))  # B x hidden_state x in_features
        x = x.flatten(start_dim=1)  # B x hidden_state * in_features
        x = self.nonlin(self.downsample(x))

        gammas = self.layer_gammas(x)
        betas = self.layer_betas(x)
        gammas = gammas + 1.
        return gammas, betas


def hmr_inference_w_context(img, gammas, betas, hmrnet, smpl_model, img_size=224):

    smpl_rotmat, smpl_shape, camera = forward_with_context(img, gammas, betas, hmrnet)
    out = smpl_model(
        betas=smpl_shape,
        body_pose=smpl_rotmat[:, 1:],
        global_orient=smpl_rotmat[:, :1],
        pose2rot=False,
    )
    j3d = out.joints
    verts3d = out.vertices

    ### align vertices with pixels
    scale, trans = camera[:, 0], camera[:, 1:]

    verts3d = verts3d * scale.view(verts3d.size(0), 1, 1)
    verts3d[:, :, 0:2] = verts3d[:, :, 0:2] + trans.view(verts3d.size(0), 1, 2)
    verts3d = (verts3d + 1) / 2 * img_size

    j3d = j3d * scale.view(j3d.size(0), 1, 1)
    j3d[:, :, 0:2] = j3d[:, :, 0:2] + trans.view(j3d.size(0), 1, 2)
    j3d = (j3d + 1) / 2 * img_size
    
    out = dict(j3d=j3d, verts3d=verts3d, camera=camera, smpl_shape=smpl_shape, smpl_rotmat=smpl_rotmat)
    return out