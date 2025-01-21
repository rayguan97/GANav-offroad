from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RELLISDataset_Group6_New(CustomDataset):
    """RELLIS dataset.

    """



    CLASSES = ("background/obstacle", "stable", "granular", "poor foothold", "high resistance", "none")

    PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]

    def __init__(self, **kwargs):
        super(RELLISDataset_Group6_New, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group6new.png',
            **kwargs)
        self.CLASSES = ("background/obstacle", "stable", "granular", "poor foothold", "high resistance", "none")
        self.PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]
