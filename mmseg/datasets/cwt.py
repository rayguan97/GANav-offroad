from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CWT_Dataset(CustomDataset):
    """cwt dataset.

    """



    CLASSES = ("flat", "bumpy", "water", "rock", "mixed", "excavator", "obstacle")

    PALETTE = [[0, 255, 0], [255, 255, 0], [255, 0, 0], [128, 0, 0], [100, 65, 0], [0, 255, 255], [0, 0, 255]]

    def __init__(self, **kwargs):
        super(CWT_Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
        self.CLASSES = ("flat", "bumpy", "water", "rock", "mixed", "excavator", "obstacle")
        self.PALETTE =[[0, 255, 0], [255, 255, 0], [255, 0, 0], [128, 0, 0], [100, 65, 0], [0, 255, 255], [0, 0, 255]]

        # assert osp.exists(self.img_dir)
