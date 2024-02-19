import argparse
import os.path as osp
import numpy as np
import mmcv
# import cv2
from PIL import Image


rellis_dir = "./data/rellis/"
annotation_folder = "annotation/"

IDs =    [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34]
# Groups = [0, 2, 2, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 4, 1, 0, 4, 3, 3]
Groups = (0, 2, 4, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 4, 1, 0, 0, 2, 3)

ID_seq = {}
ID_group = {}
for n, label in enumerate(IDs):
    ID_seq[label] = n
    ID_group[label] = Groups[n]

# 0 -- Background/obstacle: void, sky, sign, tree, pole, water, vehicle, 
#                           object, building, person, fence, barrier, puddle
# 1 -- Stable: concrete, asphalt
# 2 -- Granular: dirt, mud
# 3 -- Poor Foothold: rubble
# 4 -- High resistance: grass, bush, log

CLASSES = ("void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
            "object", "asphalt", "building", "log", "person", "fence", "bush", 
            "concrete", "barrier", "puddle", "mud", "rubble")

PALETTE = [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]


def raw_to_seq(seg):
    h, w = seg.shape
    out1 = np.zeros((h, w))
    out2 = np.zeros((h, w))
    for i in IDs:
        out1[seg==i] = ID_seq[i]
        out2[seg==i] = ID_group[i]

    return out1, out2




with open(osp.join(rellis_dir, 'train.txt'), 'r') as r:
    i = 0
    for l in r:
        print("train: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rellis_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)

        mmcv.imwrite(out1, rellis_dir + annotation_folder + l.strip() + "_orig.png")
        mmcv.imwrite(out2, rellis_dir + annotation_folder + l.strip() + "_group6new.png")

        i += 1


with open(osp.join(rellis_dir, 'val.txt'), 'r') as r:
    i = 0
    for l in r:
        print("val: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rellis_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)
        
        mmcv.imwrite(out1, rellis_dir + annotation_folder + l.strip() + "_orig.png")
        mmcv.imwrite(out2, rellis_dir + annotation_folder + l.strip() + "_group6new.png")

        i += 1



with open(osp.join(rellis_dir, 'test.txt'), 'r') as r:
    i = 0
    for l in r:
        print("test: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rellis_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)

        mmcv.imwrite(out1, rellis_dir + annotation_folder + l.strip() + "_orig.png")
        mmcv.imwrite(out2, rellis_dir + annotation_folder + l.strip() + "_group6new.png")

        i += 1




print("successful")