import argparse
import os.path as osp
import numpy as np
import mmcv
from PIL import Image


goose_dir = "./data/goose/"
annotation_dir = "labels/"

IDs =    [*range(0,64)]
Groups = (5, 5, 3, 2, 5, 2, 5, 1, 5,
          1, 5, 1, 5, 5, 5, 5, 5, 4, 2,
          5, 5, 1, 4, 1, 2, 5, 4, 5,
          5, 4, 4, 2, 5, 5, 5, 5, 5, 5, 5,
          5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
          5, 3, 4, 5, 0, 4, 5, 5, 5,
          5, 5, 5, 5, 4, 0)

ID_seq = {}
ID_group = {}
for n, label in enumerate(IDs):
    ID_seq[label] = n
    ID_group[label] = Groups[n]

# 0 background: sky
# 1 Stable: bikeway, pedesstrian_crossing, road_marking, sidewalk, asphalt,
# 2 Granular: cobble, leaves, moss, gravel, soil
# 3 Poor foothold: snow, low_grass,
# 4 High resistance: high_grass, bush, debris, crops, water, tree_root
# 5 Obstacle: everything else

CLASSES = (
    "undefined", "traffic_cone", "snow", "cobble", "obstacle", "leaves", "street_light", "bikeway", "ego_vehicle",
    "pedestrian_crossing", "road_block", "road_marking", "car", "bicycle", "person", "bus", "forest", "bush", "moss",
    "traffic_light", "motorcycle", "sidewalk", "curb", "asphalt", "gravel", "boom_barrier", "rail_track", "tree_crown",
    "tree_trunk", "debris", "crops", "soil", "rider", "animal", "truck", "on_rails", "caravan", "trailer", "building",
    "wall", "rock", "fence", "guard_rail", "bridge", "tunnel", "pole", "traffic_sign", "misc_sign", "barrier_tape",
    "kick_scooter", "low_grass", "high_grass", "scenery_vegetation", "sky", "water", "wire", "outlier", "heavy_machinery",
    "container", "hedge", "barrel", "pipe", "tree_root", "military_vehicle"
)

PALETTE = [[0, 0, 0], [255, 255, 0], [209, 87, 160], [255, 52, 255], [255, 74, 70],
           [0, 137, 65], [0, 111, 166], [163, 0, 89], [255, 219, 229], [122, 73, 0],
           [0, 0, 166], [99, 255, 172], [183, 151, 98], [0, 77, 67], [143, 176, 255],
           [153, 125, 135], [90, 0, 7], [128, 150, 147], [180, 168, 189], [27, 68, 0],
           [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128], [97, 97, 90],
           [52, 54, 45], [107, 121, 0], [0, 194, 160], [255, 170, 146], [136, 111, 76],
           [0, 134, 237], [209, 97, 0], [221, 239, 255], [0, 0, 53], [123, 79, 75],
           [161, 194, 153], [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111],
           [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191], [204, 7, 68],
           [192, 185, 178], [194, 255, 153], [0, 30, 9], [190, 196, 89], [111, 0, 98],
           [12, 189, 102], [238, 195, 255], [69, 109, 117], [183, 123, 104], [122, 135, 161],
           [255, 140, 0], [120, 141, 102], [250, 208, 159], [255, 138, 154], [209, 87, 160],
           [208, 208, 0], [221, 0, 0], [196, 164, 132], [64, 64, 64]]

def raw_to_seq(seg):
    h, w = seg.shape
    out1 = np.zeros((h, w))
    out2 = np.zeros((h, w))
    for i in IDs:
        out1[seg==i] = ID_seq[i]
        out2[seg==i] = ID_group[i]

    return out1, out2


def rewrite_set(filename):
    with open(osp.join(goose_dir, filename), 'r') as r:
        set_name = osp.splitext(filename)[0]
        i = 0
        for l in r:
            print("{}: {}".format(filename, i))
            file_client_args=dict(backend='disk')
            file_client = mmcv.FileClient(**file_client_args)
            label_path = osp.join(goose_dir, annotation_dir, set_name, l.strip() + "_labelids.png")
            img_bytes = file_client.get(label_path)
            gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
            _, out2 = raw_to_seq(gt_semantic_seg)
            mmcv.imwrite(out2, label_path.replace("labelids", "group6"))
            i += 1


rewrite_set('train.txt')
rewrite_set('val.txt')

print("successful")
