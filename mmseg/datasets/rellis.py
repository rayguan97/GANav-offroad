from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RELLISDataset(CustomDataset):
    """RELLIS dataset.
- 0: void
  1: dirt
  3: grass
  4: tree
  5: pole
  6: water
  7: sky
  8: vehicle
  9: object
  10: asphalt
  12: building
  15: log
  17: person
  18: fence
  19: bush
  23: concrete
  27: barrier
  31: puddle
  33: mud
  34: rubble
    """



    CLASSES = ("void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
            "object", "asphalt", "building", "log", "person", "fence", "bush", 
            "concrete", "barrier", "puddle", "mud", "rubble")

    PALETTE = [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]

    def __init__(self, **kwargs):
        super(RELLISDataset, self).__init__(
            img_suffix='_orig.jpg',
            seg_map_suffix='_orig.png',
            **kwargs)
        self.CLASSES = ("void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
            "object", "asphalt", "building", "log", "person", "fence", "bush", 
            "concrete", "barrier", "puddle", "mud", "rubble")
        self.PALETTE = [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]
