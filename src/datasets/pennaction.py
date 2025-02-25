from src.datasets.dataset_2d import Dataset2D

PENNACTION_DIR = "/cvlabsrc1/cvlab/dataset_PennAction/Penn_Action"

class PennAction(Dataset2D):
    # def __init__(self, seqlen, overlap=0.75, debug=True):
    def __init__(self, seqlen, overlap=0., debug=True, use_OFformat=False, videoOF_format=640, color_distort=False, output_types=None):
        db_name = 'pennaction'

        super(PennAction, self).__init__(
            seqlen = seqlen,
            folder=PENNACTION_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
            use_OFformat=use_OFformat,
            videoOF_format=videoOF_format,
            color_distort=color_distort,
            output_types=output_types
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')