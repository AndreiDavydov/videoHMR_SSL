# from src.datasets.dataset_3d import Dataset3D

# MPII3D_DIR = "/cvlabsrc1/cvlab/dataset_PennAction/Penn_Action"


# class MPII3D(Dataset3D):
#     def __init__(self, seqlen, set, overlap=0., debug=True, output_types=None):
#         db_name = 'mpii3d'

#         print('MPII3D Dataset overlap ratio: ', overlap)
#         super(MPII3D, self).__init__(
#             seqlen=seqlen,
#             set=set, # can be "train" or "val"
#             folder=MPII3D_DIR,
#             dataset_name=db_name,
#             debug=debug,
#             overlap=overlap,
#             output_types=output_types
#         )
#         print(f'{db_name} - number of dataset objects {self.__len__()}')