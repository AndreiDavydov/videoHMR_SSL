# taken from https://github.com/hongsukchoi/TCMR_RELEASE/blob/master/lib/dataset/threedpw.py

import os
import shutil

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
from src.datasets.dataset_3d import Dataset3D

# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

THREEDPW_DIR = "manifold://xr_body/tree/personal/andreydavydov/3dpw/"
THREEDPW_DIR_LOCAL = "/tmp/3dpw/"


class ThreeDPW(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.75, debug=False, copy_to_local=True):
        # if copy_to_local:
        #     ### copy zip to local
        #     images_zip = pathmgr.get_local_path(
        #         os.path.join(THREEDPW_DIR, "imageFiles.zip")
        #     )
        #     ### unarchive
        #     if not os.path.exists(os.path.join(THREEDPW_DIR_LOCAL, "imageFiles")):
        #         os.makedirs(THREEDPW_DIR_LOCAL)
        #         shutil.unpack_archive(images_zip, THREEDPW_DIR_LOCAL)

        #     image_files_dir = THREEDPW_DIR_LOCAL
        # else:
        #     image_files_dir = THREEDPW_DIR
        image_files_dir = THREEDPW_DIR

        db_name = "3dpw"

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        is_train = False
        overlap = overlap if is_train else 0.0
        print("3DPW Dataset overlap ratio: ", overlap)
        super(ThreeDPW, self).__init__(
            set=set,
            folder=image_files_dir,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f"{db_name} - number of dataset objects {self.__len__()}")
