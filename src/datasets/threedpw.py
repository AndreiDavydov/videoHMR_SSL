# taken from https://github.com/hongsukchoi/TCMR_RELEASE/blob/master/lib/dataset/threedpw.py

from src.datasets.dataset_3d import Dataset3D

THREEDPW_DIR = "/cvlabdata2/home/davydov/videoHMR_SSL/data/3dpw/3dpw_original"


class ThreeDPW(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.0, debug=True, use_OFformat=False, videoOF_format=640, color_distort=False, output_types=None):
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
            use_OFformat=use_OFformat,
            videoOF_format=videoOF_format,
            color_distort=color_distort,
            output_types=output_types
        )
        print(f"{db_name} - number of dataset objects {self.__len__()}")
