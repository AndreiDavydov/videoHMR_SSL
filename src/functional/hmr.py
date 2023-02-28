import torch


def hmr_inference(img, hmrnet, smpl_model, img_size=224):
    smpl_rotmat, smpl_shape, camera = hmrnet(img)
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
    out = dict(j3d=j3d, verts3d=verts3d, camera=camera)
    return out