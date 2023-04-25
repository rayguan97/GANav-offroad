from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RELLISDataset_Group4(CustomDataset):
    """RELLIS dataset.

    """



    CLASSES = ("background", "L1 (Smooth)", "non-Nav (Forbidden)", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]

    def __init__(self, **kwargs):
        super(RELLISDataset_Group4, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group4.png',
            **kwargs)
        self.CLASSES = ("background", "L1 (Smooth)", "non-Nav (Forbidden)", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]
