import logging
import os
import os.path as osp

import joblib
import json
import numpy as np
import torch
from torch.utils.data import Dataset

THREEDPW_DIR = "/cvlabdata2/home/davydov/videoHMR_SSL/data/3dpw/3dpw_processed/"

MPII3D_DIR = "/cvlabdata2/home/davydov/videoHMR_SSL/data/mpii3d/"

H36M_DIR = "/cvlabdata2/home/davydov/videoHMR_SSL/data/h36m/"
ACTIONS = ['Directions','Discussion','Eating','Greeting',
             'Phoning','Posing','Purchases','Sitting','SittingDown',
             'Smoking','TakingPhoto', 'Waiting','WalkTogether','Walking','WalkingDog',]
CAMERAS_CODES = ['54138969', '55011271', '58860488', '60457274']


from skimage.util.shape import view_as_windows

import src.utils.img_utils_datasets_3d as utils

logger = logging.getLogger(__name__)


def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices


def get_names(data_type):
    if data_type == "common":
        return COMMON_JOINT_NAMES
    if data_type == "spin":
        return SPIN_JOINT_NAMES
    if data_type == "pennaction":
        return PENNACTION_JOINT_NAMES


def convert_kps(joints2d, src, dst):
    src_names = get_names(src)
    dst_names = get_names(dst)

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), 3))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d


class Dataset3D(Dataset):
    def __init__(
        self, set, seqlen, overlap=0.0, folder=None, dataset_name=None, debug=False, use_OFformat=False, videoOF_format=640, output_types=None
    ):
        self.folder = folder
        self.set = set
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.stride = int(seqlen * (1 - overlap))
        self.debug = debug
        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db["vid_name"], self.seqlen, self.stride)

        self.output_types = output_types

        self.use_OFformat = use_OFformat
        self.videoOF_format = videoOF_format

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        if self.dataset_name == "3dpw":
            db_file = osp.join(THREEDPW_DIR, f"{self.set}/{self.dataset_name}_{self.set}_db.pt")
        elif self.dataset_name == "mpii3d":
            db_file = osp.join(MPII3D_DIR, f"{self.dataset_name}_{self.set}_scale12_db.pt")
        elif self.dataset_name == 'h36m':
            db_file = osp.join(H36M_DIR, f'{self.dataset_name}_{self.set}_25fps_db.pt')

        # db_file = pathmgr.get_local_path(db_file)
        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f"{db_file} do not exists")

        if self.dataset_name == "h36m":
            ### mask samples, some of them are broken
            with open(f"{H36M_DIR}correct_mask_h36m_{self.set}.json", "r") as f:
                mask = json.load(f)
            for k in db:
                db[k] = db[k][mask]

            ### take every second frame, otherwise frames are too similar
            for k in db:
                db[k] = db[k][::2]

            ### load video subacts
            with open(f"{H36M_DIR}video_subacts.json", "r") as f:
                self.video_subacts = json.load(f)

        print(f"Loaded {self.dataset_name} dataset from {db_file}")
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        is_train = self.set == "train"

        if self.dataset_name == "3dpw":
            kp_2d = convert_kps(
                self.db["joints2D"][start_index : end_index + 1],
                src="common",
                dst="spin",
            )
            kp_3d = self.db["joints3D"][start_index : end_index + 1]

        elif self.dataset_name == 'h36m':
            kp_2d = self.db['joints2D'][start_index:end_index+1]
            if is_train:
                kp_3d = self.db['joints3D'][start_index:end_index+1]
            else:
                kp_3d = convert_kps(self.db['joints3D'][start_index:end_index+1], src='spin', dst='common')

        kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)

        if is_train:
            nj = 49
        else:
            if self.dataset_name == 'mpii3d':
                nj = 17
            else:
                nj = 14

        kp_3d_tensor = np.zeros((self.seqlen, nj, 3), dtype=np.float16)

        if self.dataset_name == "3dpw":
            pose = self.db["pose"][start_index : end_index + 1]
            shape = self.db["shape"][start_index : end_index + 1]
            w_smpl = torch.ones(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()
        elif self.dataset_name == 'h36m':
            if not is_train:
                pose = np.zeros((kp_2d.shape[0], 72))
                shape = np.zeros((kp_2d.shape[0], 10))
                w_smpl = torch.zeros(self.seqlen).float()
                w_3d = torch.ones(self.seqlen).float()
            else:
                pose = self.db['pose'][start_index:end_index+1]
                shape = self.db['shape'][start_index:end_index+1]
                # SMPL parameters obtained by NeuralAnnot is released now! - 06/17/2022
                w_smpl = torch.ones(self.seqlen).float()
                w_3d = torch.ones(self.seqlen).float()

        bbox = self.db["bbox"][start_index : end_index + 1]
        input = torch.from_numpy(
            self.db["features"][start_index : end_index + 1]
        ).float()

        theta_tensor = np.zeros((self.seqlen, 85), dtype=np.float16)

        for idx in range(self.seqlen):
            # crop image and transform 2d keypoints
            kp_2d[idx, :, :2], trans = utils.transform_keypoints(
                kp_2d=kp_2d[idx, :, :2],
                center_x=bbox[idx, 0],
                center_y=bbox[idx, 1],
                width=bbox[idx, 2],
                height=bbox[idx, 3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx, :, :2] = utils.normalize_2d_kp(kp_2d[idx, :, :2], 224)

            # theta shape (85,)
            theta = np.concatenate(
                (np.array([1.0, 0.0, 0.0]), pose[idx], shape[idx]), axis=0
            )

            kp_2d_tensor[idx] = kp_2d[idx]
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]

        target = {
            "features": input,
            "theta": torch.from_numpy(theta_tensor).float(),  # camera, pose and shape
            "kp_2d": torch.from_numpy(kp_2d_tensor).float(),  # 2D keypoints transformed according to bbox cropping
            "kp_3d": torch.from_numpy(kp_3d_tensor).float(),  # 3D keypoints
            "w_smpl": w_smpl,
            "w_3d": w_3d,
        }

        if self.dataset_name == "3dpw": # and not is_train:
            vn = self.db["vid_name"][start_index : end_index + 1]
            fi = self.db["frame_id"][start_index : end_index + 1]
            target["instance_id"] = [f"{v}/{f}" for v, f in zip(vn, fi)]
            
            # if self.dataset_name == '3dpw' and not self.is_train:
            # target['imgname'] = self.db['img_name'][start_index:end_index+1].tolist()
            # target['imgname'] = np.array(target['imgname'])
            # print(target['imgname'].dtype)
            # target['center'] = self.db['bbox'][start_index:end_index+1, :2]
            # target['valid'] = torch.from_numpy(self.db['valid'][start_index:end_index+1])

        elif self.dataset_name == "h36m": # and not is_train:
            imgname = (self.db["img_name"][start_index : end_index + 1]).tolist()
            target['imgname_old'] = imgname.copy()
            target['imgname'] = []
            for imgname_ in imgname:
                subj, action, subact, cam, frame = parse_h36m_imgname(imgname_, self.video_subacts)
                imgname_ = combine_imgname(subj, action, subact, cam, frame, root=self.folder)
                target['imgname'].append(imgname_)

        ### add attr-s to reconstruct GT smpl bodies aligned with bbox image
        target["bbox"] = bbox

        if self.debug:

            if self.dataset_name == "mpii3d":
                video_files = self.db["img_name"][start_index : end_index + 1]
            elif self.dataset_name == "h36m":
                video_files = target['imgname'].copy()
            else:
                vid_name = self.db["vid_name"][start_index]
                vid_name = "_".join(vid_name.split("_")[:-1])
                f = osp.join(self.folder, "imageFiles", vid_name)
                video_file_list = [
                    osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith(".jpg")
                ]
                frame_idxs = self.db["frame_id"][start_index : end_index + 1]
                # print(f, frame_idxs)
                video_files = [video_file_list[i] for i in frame_idxs]
                target['imgname'] = video_files.copy()
            
            video = torch.cat([
                    utils.get_single_image_crop(image, bbox).unsqueeze(0)
                    for image, bbox in zip(video_files, bbox)],dim=0,)

            target["video"] = video

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


def parse_h36m_imgname(imgname, video_subacts):
    subj = int(imgname.split("_act_")[0].split("s_")[-1])
    subj = f"S{subj}"

    action = int(imgname.split("_act_")[-1].split("_subact_")[0]) - 2
    action = ACTIONS[action]

    subact = int(imgname.split("_subact_")[-1].split("_ca_")[0])
    subact = get_subact(subj, action, subact, video_subacts)

    cam = int(imgname.split("_ca_")[-1][:2]) - 1
    cam = CAMERAS_CODES[cam]
    
    frame = int(imgname.split("_")[-1][:-4])

    return subj, action, subact, cam, frame 


def get_subact(subj, action, subact_id, video_subacts):
    """
    subact_id : always either 1 or 2
    video_subacts : dict of existing (videos) subacts of H36M
    """
    return video_subacts[subj][action][subact_id-1]
    

def combine_imgname(subj, action, subact, cam, frame, root):
    line = os.path.join(root, subj, "Images", f"{action}{subact}.{cam}_{frame:012}.jpg")
    return line
    

COMMON_JOINT_NAMES = [
    "rankle",  # 0  "lankle",    # 0
    "rknee",  # 1  "lknee",     # 1
    "rhip",  # 2  "lhip",      # 2
    "lhip",  # 3  "rhip",      # 3
    "lknee",  # 4  "rknee",     # 4
    "lankle",  # 5  "rankle",    # 5
    "rwrist",  # 6  "lwrist",    # 6
    "relbow",  # 7  "lelbow",    # 7
    "rshoulder",  # 8  "lshoulder", # 8
    "lshoulder",  # 9  "rshoulder", # 9
    "lelbow",  # 10  "relbow",    # 10
    "lwrist",  # 11  "rwrist",    # 11
    "neck",  # 12  "neck",      # 12
    "headtop",  # 13  "headtop",   # 13
]

SPIN_JOINT_NAMES = [
    "OP Nose",  # 0
    "OP Neck",  # 1
    "OP RShoulder",  # 2
    "OP RElbow",  # 3
    "OP RWrist",  # 4
    "OP LShoulder",  # 5
    "OP LElbow",  # 6
    "OP LWrist",  # 7
    "OP MidHip",  # 8
    "OP RHip",  # 9
    "OP RKnee",  # 10
    "OP RAnkle",  # 11
    "OP LHip",  # 12
    "OP LKnee",  # 13
    "OP LAnkle",  # 14
    "OP REye",  # 15
    "OP LEye",  # 16
    "OP REar",  # 17
    "OP LEar",  # 18
    "OP LBigToe",  # 19
    "OP LSmallToe",  # 20
    "OP LHeel",  # 21
    "OP RBigToe",  # 22
    "OP RSmallToe",  # 23
    "OP RHeel",  # 24
    "rankle",  # 25
    "rknee",  # 26
    "rhip",  # 27
    "lhip",  # 28
    "lknee",  # 29
    "lankle",  # 30
    "rwrist",  # 31
    "relbow",  # 32
    "rshoulder",  # 33
    "lshoulder",  # 34
    "lelbow",  # 35
    "lwrist",  # 36
    "neck",  # 37
    "headtop",  # 38
    "hip",  # 39 'Pelvis (MPII)', # 39
    "thorax",  # 40 'Thorax (MPII)', # 40
    "Spine (H36M)",  # 41
    "Jaw (H36M)",  # 42
    "Head (H36M)",  # 43
    "nose",  # 44
    "leye",  # 45 'Left Eye', # 45
    "reye",  # 46 'Right Eye', # 46
    "lear",  # 47 'Left Ear', # 47
    "rear",  # 48 'Right Ear', # 48
]

PENNACTION_JOINT_NAMES = [
    "headtop",   # 0
    "lshoulder", # 1
    "rshoulder", # 2
    "lelbow",    # 3
    "relbow",    # 4
    "lwrist",    # 5
    "rwrist",    # 6
    "lhip" ,     # 7
    "rhip" ,     # 8
    "lknee",     # 9
    "rknee" ,    # 10
    "lankle",    # 11
    "rankle"     # 12
   ]