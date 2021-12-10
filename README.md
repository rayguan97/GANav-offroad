# GANav: Group-wise Attention Network for Classifying Navigable Regions in Unstructured Outdoor Environments

## Updates:
12-9-2021: updated ros-support for GANav [Here](https://github.com/rayguan97/GANav-offroad/tree/main/ros_support).

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/rayguan97/GANav-offroad/blob/master/LICENSE)

This is the code base for:

[GANav: Group-wise Attention Network for Classifying Navigable Regions in Unstructured Outdoor Environments](https://gamma.umd.edu/offroad).
<br> Tianrui Guan, Divya Kothandaraman, Rohan Chandra, Dinesh Manocha

<img src="./resources/video.gif" width="560">

If you find this project useful in your research, please cite our work:

```latex
@misc{guan2021ganav,
      title={GANav: Group-wise Attention Network for Classifying Navigable Regions in Unstructured Outdoor Environments}, 
      author={Tianrui Guan and Divya Kothandaraman and Rohan Chandra and Adarsh Jagan Sathyamoorthy and Dinesh Manocha},
      year={2021},
      eprint={2103.04233},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

Our video can be found [here](https://www.youtube.com/watch?v=wy4k7Oz1HHM).

# Introduction

<img src="./resources/cover.png" width="700">


We present a new learning-based method for identifying safe and navigable regions in off-road terrains and unstructured environments from RGB images. Our approach consists of classifying groups of terrain classes based on their navigability levels using coarse-grained semantic segmentation. We propose a bottleneck transformer-based deep neural network architecture that uses a novel group-wise attention mechanism to distinguish between navigability levels of different terrains. Our group-wise attention heads enable the network to explicitly focus on the different groups and improve the accuracy. 

We show through extensive evaluations on the RUGD and RELLIS-3D datasets that our learning algorithm improves the accuracy of visual perception in off-road terrains for navigation. We compare our approach with prior work on these datasets and achieve an improvement over the state-of-the-art mIoU by 6.74-39.1% on RUGD and 3.82-10.64% on RELLIS-3D.

# Environment


### Step 1: Create Conda Environment

```
conda create -n ganav python=3.7 -y
conda activate ganav
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
```

### Step 2: Installing MMCV

```
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```
Note: Make sure you mmcv version is compatible with your pytorch and cuda version.

### Step 3: Installing GANav
```
git clone https://github.com/rayguan97/GANav-offroad.git
cd GANav-offroad
pip install einops
pip install -e . 
```


# Get Started

In this section, we explain the data generation process and how to train and test our network.

## Data Processing

To be able to run our network, please follow those steps for generating processed data.

### Dataset Download: 

Please go to [RUGD](http://rugd.vision/) and [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D/blob/main/README.md#annotated-data) (we use the ID annotation instead of color annotation for RELLIS-3D) officail website to download their data. Please structure the downloaded datain as follows:

```
GANav
├── data
│   ├── rellis
│   │   │── test.txt
│   │   │── train.txt
│   │   │── val.txt
│   │   │── annotation
│   │   │   ├── 00000 & 00001 & 00002 & 00003 & 00004 
│   │   │── image
│   │   │   ├── 00000 & 00001 & 00002 & 00003 & 00004 
│   ├── rugd
│   │   │── test_ours.txt
│   │   │── test.txt
│   │   │── train_ours.txt
│   │   │── train.txt
│   │   │── val_ours.txt
│   │   │── val.txt
│   │   │── RUGD_annotations
│   │   │   ├── creek & park-1/2/8 & trail-(1 & 3-7 & 9-15) & village
│   │   │── RUGD_frames-with-annotations
│   │   │   ├── creek & park-1/2/8 & trail-(1 & 3-7 & 9-15) & village
├── configs
├── tools
...
```

### Dataset Processing: 

In this step, we need to process the groundtruth labels, as well as generating the grouped labels.

For RELLIS-3D dataset, run:

   ```
   python ./tools/convert_datasets/rellis_relabel[x].py
   ``` 

For RUGD dataset, run:

   ```
   python ./tools/convert_datasets/rugd_relabel[x].py
   ``` 

Replease [x] with 4 or 6, to generated data with 4 annotation groups or 6 annotation groups.

## Training

To train a model on RUGD datasets with our methods on 6 groups:
```
python ./tools/train.py ./configs/ours/ganav_group6_rugd.py
```

Please modify `./configs/ours/*` to play with your model and read `./tools/train.py` for more details about training options.

## Testing

An example to evaluate our method with 6 groups on RUGD datasets with mIoU metrics:

```
python ./tools/test.py ./trained_models/rugd_group6/ganav_rugd.py \
          ./trained_models/rugd_group6/ganav_rugd.pth --eval=mIoU
```
Please read `./tools/test.py` for more details.

To repreduce the papers results, please refer `./trained_models` folder. Please download the trained model [here](https://drive.google.com/drive/folders/1PYn_kT0zBGOIRSaO_5Jivaq3itrShiPT?usp=sharing).



# License

This project is released under the [Apache 2.0 license](LICENSE).

# Acknowledgement

The source code of GANav is heavily based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). 

