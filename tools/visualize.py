from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import torch
import mmcv
import numpy as np
from shutil import copyfile
import os
import datetime


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('-p', default=".", type=str)
    parser.add_argument('-d', action='store_true')
    parser.add_argument('-s', default="./vis.png", type=str)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='rugd_group',
        help='Color palette used for segmentation map')
    args = parser.parse_args()


    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    img = mmcv.imread(args.img)

    result = inference_segmentor(model, img)

    show_result_pyplot(model, args.img , result, get_palette(args.palette), save_dir="./vis/pred.png", display=args.d, seg_only=True)




if __name__ == '__main__':
    main()
