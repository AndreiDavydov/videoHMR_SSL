from src.datasets.dataset_3d import Dataset3D

H36M_DIR = '/cvlabsrc1/cvlab/dataset_H36M_SMPL/Human36m/OpenPose'


class Human36M(Dataset3D):
    def __init__(self, seqlen, set, overlap=0., debug=True, output_types=None):
        db_name = 'h36m'

        print('Human36M Dataset overlap ratio: ', overlap)
        super(Human36M, self).__init__(
            seqlen=seqlen,
            set=set, # can be "train" or "test"
            folder=H36M_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
            output_types=output_types
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')


