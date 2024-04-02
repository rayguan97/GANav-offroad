from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GOOSEDataset_Group6(CustomDataset):

    CLASSES = ("background", "stable", "granular", "poor foothold", "high resistance", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ], [ 255, 0, 0 ],[  0, 0, 128]]

    def __init__(self, **kwargs):
        super(GOOSEDataset_Group6, self).__init__(
            img_suffix='_windshield_vis.png',
            seg_map_suffix='_group6.png',
            **kwargs)
        self.CLASSES = ("background", "stable", "granular", "poor foothold", "high resistance", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ], [ 255, 0, 0 ],[  0, 0, 128]]
