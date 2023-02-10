import random
from src.datasets.coco_eft import COCO_EFT

class COCO_EFT_Fractional(COCO_EFT):
    def __init__(
        self,
        fraction=100, # fraction of data to use (in %)
        **kwargs
    ):
        super(COCO_EFT_Fractional, self).__init__(**kwargs)

        ids = list(range(len(self.kpts)))
        
        r = int( fraction/100 * len(ids) )
        random.seed(0)
        random.shuffle(ids)
        self.ids = ids[:r]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        new_idx = self.ids[idx]
        out = super(COCO_EFT_Fractional, self).__getitem__(new_idx)
        return out


if __name__ == "__main__":
    dset = COCO_EFT_Fractional()
    print(len(dset))
    print(dset[0].keys())
