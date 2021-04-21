from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RUGDDataset(CustomDataset):
    """RUGD dataset.

0 void 0 0 0
1 dirt 108 64 20
2 sand 255 229 204
3 grass 0 102 0
4 tree 0 255 0
5 pole 0 153 153
6 water 0 128 255
7 sky 0 0 255
8 vehicle 255 255 0
9 container/generic-object 255 0 127
10 asphalt 64 64 64
11 gravel 255 128 0
12 building 255 0 0
13 mulch 153 76 0
14 rock-bed 102 102 0
15 log 102 0 0
16 bicycle 0 255 128
17 person 204 153 255
18 fence 102 0 204
19 bush 255 153 204
20 sign 0 102 102
21 rock 153 204 255
22 bridge 102 255 255
23 concrete 101 101 11
24 picnic-table 114 85 47

    """



    CLASSES = ("dirt", "sand", "grass", "tree", "pole", "water", "sky", 
                "vehicle", "container/generic-object", "asphalt", "gravel", 
                "building", "mulch", "rock-bed", "log", "bicycle", "person", 
                "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table")

    PALETTE = [[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
                [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
                [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
                [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
                [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ] ]


    def __init__(self, **kwargs):
        super(RUGDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_normal.png',
            **kwargs)
        self.CLASSES = ("dirt", "sand", "grass", "tree", "pole", "water", "sky", 
                "vehicle", "container/generic-object", "asphalt", "gravel", 
                "building", "mulch", "rock-bed", "log", "bicycle", "person", 
                "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table")
        self.PALETTE = [[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
                [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
                [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
                [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
                [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ] ]
