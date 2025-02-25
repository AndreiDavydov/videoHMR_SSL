from src.datasets.dataset_3d import Dataset3D

H36M_DIR = '/cvlabsrc1/cvlab/dataset_H36M_SMPL/Human36m/OpenPose'


class Human36M(Dataset3D):
    def __init__(self, seqlen, set, overlap=0., debug=True, use_OFformat=False, videoOF_format=640, color_distort=False, output_types=None):
        db_name = 'h36m'

        print('Human36M Dataset overlap ratio: ', overlap)
        super(Human36M, self).__init__(
            seqlen=seqlen,
            set=set, # can be "train" or "test"
            folder=H36M_DIR,
            dataset_name=db_name,
            debug=debug,
            use_OFformat=use_OFformat,
            videoOF_format=videoOF_format,
            overlap=overlap,
            color_distort=color_distort,
            output_types=output_types
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')


