from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RELLISDataset_Group6(CustomDataset):
    """RELLIS dataset.

    """



    CLASSES = ("background", "L1 (Smooth)", "L2 (Rough)", "L3 (Bumpy)", "non-Nav (Forbidden)", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]


    def __init__(self, **kwargs):
        super(RELLISDataset_Group6, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group6.png',
            **kwargs)
        self.CLASSES = ("background", "L1 (Smooth)", "L2 (Rough)", "L3 (Bumpy)", "non-Nav (Forbidden)", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]
