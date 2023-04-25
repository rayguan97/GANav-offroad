from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RUGDDataset_Group4(CustomDataset):
    """RUGD dataset.

    """



    CLASSES = ("background", "L1 (Smooth)", "non-Nav (Forbidden)", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]

    def __init__(self, **kwargs):
        super(RUGDDataset_Group4, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_group4.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "L1 (Smooth)", "non-Nav (Forbidden)", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]
        # assert osp.exists(self.img_dir)
