import argparse
import os.path as osp
import numpy as np
import mmcv
# import cv2
from PIL import Image


rudg_dir = "./data/rugd/"
annotation_folder = "RUGD_annotations/"

CLASSES = ("dirt", "sand", "grass", "tree", "pole", "water", "sky", 
        "vehicle", "container/generic-object", "asphalt", "gravel", 
        "building", "mulch", "rock-bed", "log", "bicycle", "person", 
        "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table")

PALETTE = [ [ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
            [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
            [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
            [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ] ]

Groups = [2, 2, 2, 5, 5, 4, 0, 5, 5, 1, 2, 5, 2, 3, 5, 5, 5, 5, 5, 0, 3, 5, 1, 5]


# 0 -- Background: void, sky, sign
# 1 -- Level1 (smooth) - Navigable: concrete, asphalt
# 2 -- Level2 (rough) - Navigable: gravel, grass, dirt, sand, mulch
# 3 -- Level3 (bumpy) - Navigable: Rock, Rock-bed
# 4 -- Non-Navigable (forbidden) - water
# 5 -- Obstacle - tree, pole, vehicle, container/generic-object, building, log, 
#                 bicycle(could be removed), person, fence, bush, picnic-table, bridge,

color_id = {tuple(c):i for i, c in enumerate(PALETTE)}
color_id[tuple([0, 0, 0])] = 255

def rgb2mask(img):
    # assert len(img) == 3
    h, w, c = img.shape
    out = np.ones((h, w, c)) * 255
    for i in range(h):
        for j in range(w):
            if tuple(img[i, j]) in color_id:
                out[i][j] = color_id[tuple(img[i, j])]
            else:
                print("unknown color, exiting...")
                exit(0)
    return out


def raw_to_seq(seg):
    h, w = seg.shape
    out = np.zeros((h, w))
    for i in range(len(Groups)):
        out[seg==i] = Groups[i]

    out[seg==255] = 0
    return out


with open(osp.join(rudg_dir, 'train_ours.txt'), 'r') as r:
    i = 0
    for l in r:
        print("train: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rudg_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
        out = rgb2mask(gt_semantic_seg)
        out = out[:, :, 0]
        mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
        out2 = raw_to_seq(out)
        mmcv.imwrite(out2, rudg_dir + annotation_folder + l.strip() + "_group6.png")

        i += 1


with open(osp.join(rudg_dir, 'val_ours.txt'), 'r') as r:
    i = 0
    for l in r:
        print("val: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rudg_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
        out = rgb2mask(gt_semantic_seg)
        out = out[:, :, 0]
        mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
        out2 = raw_to_seq(out)
        mmcv.imwrite(out2, rudg_dir + annotation_folder + l.strip() + "_group6.png")

        i += 1



with open(osp.join(rudg_dir, 'test_ours.txt'), 'r') as r:
    i = 0
    for l in r:
        print("test: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rudg_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
        out = rgb2mask(gt_semantic_seg)
        out = out[:, :, 0]
        mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
        out2 = raw_to_seq(out)
        mmcv.imwrite(out2, rudg_dir + annotation_folder + l.strip() + "_group6.png")

        i += 1



print("successful")