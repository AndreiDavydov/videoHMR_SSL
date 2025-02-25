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

    j3d = j3d * scale.view(j3d.size(0), 1, 1)
    j3d[:, :, 0:2] = j3d[:, :, 0:2] + trans.view(j3d.size(0), 1, 2)
    j3d = (j3d + 1) / 2 * img_size
    
    out = dict(j3d=j3d, verts3d=verts3d, camera=camera, smpl_shape=smpl_shape, smpl_rotmat=smpl_rotmat)
    return out