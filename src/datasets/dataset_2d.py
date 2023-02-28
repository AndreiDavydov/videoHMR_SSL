# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import random
import logging
import numpy as np
import os.path as osp
import joblib

from torch.utils.data import Dataset

from src.datasets.dataset_3d import convert_kps, split_into_chunks
import src.utils.img_utils_datasets_3d as utils


logger = logging.getLogger(__name__)


class Dataset2D(Dataset):
    def __init__(self, seqlen, overlap=0., folder=None, dataset_name=None, debug=True, use_OFformat=False, videoOF_format=640, output_types=None):

        self.set = 'train'
        self.folder = folder
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.mid_frame = int(seqlen/2)
        self.stride = int(seqlen * (1-overlap) + 0.5)
        self.debug = debug
        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], seqlen, self.stride)

        self.output_types = output_types

        self.use_OFformat = use_OFformat
        self.videoOF_format = videoOF_format

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        if self.dataset_name == "pennaction":
            db_file = osp.join(self.folder, '../pennaction_train_scale12_db.pt')
        else:
            raise NotImplementedError(f"Loading of {self.dataset_name} is not yet implemented")

        # db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{set}_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')

        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
        if self.dataset_name != 'posetrack':
            kp_2d = convert_kps(kp_2d, src=self.dataset_name, dst='spin')
        kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)

        bbox = self.get_sequence(start_index, end_index, self.db['bbox'])

        if self.dataset_name == "pennaction":
            for i in range(self.seqlen):
                ### fix unequal bbox H, W
                bbox[i,2] = bbox[i,3] = max(bbox[i,2], bbox[i,3])

                ### bbox centers are not centered in the body
                ### fix it by finding centers given the kp_2d coordinates means
                mask_ = kp_2d[i][:,2] > 0.
                if mask_.sum() > 0: # kpts is not empty
                    c_x = kp_2d[i][:,0][mask_].mean()
                    c_y = kp_2d[i][:,1][mask_].mean()
                    bbox[i,0] = c_x
                    bbox[i,1] = c_y

                    ### sometimes bbox is too wide compared to the size of the body
                    delta_x = kp_2d[i][:,0][mask_].max() - kp_2d[i][:,0][mask_].min()
                    delta_y = kp_2d[i][:,1][mask_].max() - kp_2d[i][:,1][mask_].min()
                    approp_scale = 1.3
                    tol = 0.0001
                    x_scale = bbox[i,2] / (delta_x + tol)
                    y_scale = bbox[i,3] / (delta_y + tol)
                    if x_scale > approp_scale and y_scale > approp_scale:
                        xy_scale = min(x_scale, y_scale)
                        bbox[i,2] = bbox[i,2] / xy_scale * approp_scale
                        bbox[i,3] = bbox[i,3] / xy_scale * approp_scale


        ### keep original bbox-s, before cropping and scaling                
        kp_2d_orig = kp_2d.copy() 

        input = torch.from_numpy(self.get_sequence(start_index, end_index, self.db['features'])).float()

        for idx in range(self.seqlen):
            # crop image and transform 2d keypoints
            kp_2d[idx,:,:2], trans = utils.transform_keypoints(
                kp_2d=kp_2d[idx,:,:2],
                center_x=bbox[idx,0],
                center_y=bbox[idx,1],
                width=bbox[idx,2],
                height=bbox[idx,3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx,:,:2] = utils.normalize_2d_kp(kp_2d[idx,:,:2], 224)
            kp_2d_tensor[idx] = kp_2d[idx]

        vid_name = self.get_sequence(start_index, end_index, self.db['vid_name'])
        frame_id = self.get_sequence(start_index, end_index, self.db['img_name']).astype(str)
        instance_id = np.array([v+f for v,f in zip(vid_name, frame_id)])

        # video = torch.cat(
        #     [utils.get_single_image_crop(image, None, bbox, scale=1.2).unsqueeze(0) for idx, (image, bbox) in
        #      enumerate(zip(frame_id, bbox))], dim=0
        # )
        target = {
            "kp_2d_orig" : kp_2d_orig,
            'features': input,
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),
            # 'kp_2d': torch.from_numpy(kp_2d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1), # 2D keypoints transformed according to bbox cropping
            'instance_id': instance_id,
            'bbox' : bbox
        }

        if self.debug:

            vid_name = self.db['vid_name'][start_index]

            if self.dataset_name == 'pennaction':
                vid_folder = "frames"
                vid_name = vid_name.split('/')[-1].split('.')[0]
                img_id = "img_name"
            elif self.dataset_name == 'posetrack':
                vid_folder = osp.join('images', vid_name.split('/')[-2])
                vid_name = vid_name.split('/')[-1].split('.')[0]
                img_id = "img_name"
            else:
                vid_name = '_'.join(vid_name.split('_')[:-1])
                vid_folder = 'imageFiles'
                img_id= 'frame_id'
            f = osp.join(self.folder, vid_folder, vid_name)
            video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
            frame_idxs = self.get_sequence(start_index, end_index, self.db[img_id])
            
            if self.dataset_name == 'pennaction':
                frame_idxs = [frame_id.split("/")[-1] for frame_id in frame_idxs]
                video_files = [osp.join(f, frame_id) for frame_id in frame_idxs]
                target['imgname'] = video_files.copy()
            else:
                raise NotImplementedError
            # if self.dataset_name == 'pennaction' or self.dataset_name == 'posetrack':
            #     video = frame_idxs
            # else:
            #     video = [video_file_list[i] for i in frame_idxs]
            video = torch.cat(
                [utils.get_single_image_crop(image, bbox).unsqueeze(0) for image, bbox in zip(video_files, bbox)], dim=0
            )

            target['video'] = video

            if self.use_OFformat:
                videoOF = torch.cat([
                    utils.get_single_image_crop(
                        image, 
                        bbox, 
                        patch_width=self.videoOF_format, 
                        patch_height=self.videoOF_format).unsqueeze(0)
                    for image, bbox in zip(video_files, bbox)],dim=0,)

                target["videoOF"] = videoOF
                
        target_up = {}
        if self.output_types is not None:
            for output_type in self.output_types:
                target_up[output_type] = target[output_type]
            return target_up
        
        return target

