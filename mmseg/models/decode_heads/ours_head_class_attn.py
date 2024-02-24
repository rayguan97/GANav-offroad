import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import math
import torch.utils.model_zoo as model_zoo
import numpy as np

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.transnet import Attention
from ..builder import build_loss
from mmcv.runner import auto_fp16, force_fp32
from ..losses import accuracy
from mmseg.core import eval_metrics

try:
    from mmcv.ops import PSAMask
except ModuleNotFoundError:
    PSAMask = None

@HEADS.register_module()
class OursHeadClassAtt(BaseDecodeHead):
    """Point-wise Spatial Attention Network for Scene Parsing.

    This head is the implementation of `PSANet
    <https://hszhao.github.io/papers/eccv18_psanet.pdf>`_.

    Args:
        mask_size (tuple[int]): The PSA mask size. It usually equals input
            size.
        psa_type (str): The type of psa module. Options are 'collect',
            'distribute', 'bi-direction'. Default: 'bi-direction'
        compact (bool): Whether use compact map for 'collect' mode.
            Default: True.
        shrink_factor (int): The downsample factors of psa mask. Default: 2.
        normalization_factor (float): The normalize factor of attention.
        psa_softmax (bool): Whether use softmax for attention.
    """

    def __init__(self,
                 mask_size,
                 psa_type='bi-direction',
                 compact=False,
                 shrink_factor=2,
                 normalization_factor=1.0,
                 psa_softmax=True,
                 img_size=(300, 375),
                 strides=(1, 2, 2, 2),
                 size_index=0,
                 **kwargs):
        if PSAMask is None:
            raise RuntimeError('Please install mmcv-full for PSAMask ops')
        super(OursHeadClassAtt, self).__init__(**kwargs)
        assert psa_type in ['collect', 'distribute', 'bi-direction']
        self.psa_type = psa_type
        self.compact = compact
        self.shrink_factor = shrink_factor
        self.mask_size = mask_size
        mask_h, mask_w = mask_size
        self.psa_softmax = psa_softmax
        if normalization_factor is None:
            normalization_factor = mask_h * mask_w
        self.normalization_factor = normalization_factor
        h, w = img_size
        # from IPython import embed;embed()
        for s in [2, 2] + [x for x in strides if x != 1]:
            if h % 2 != 0:
                h += 1
            if w % 2 != 0:
                w += 1   
            h /= s 
            w /= s 
        self.f_h = int(h)
        self.f_w = int(w)
        self.attn_loss = build_loss(dict(
                     type='ClassAttCrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0, num_classes=self.num_classes))
        self.size_index = size_index
        self.attn = Attention(dim=self.channels, fmap_size=(self.f_h, self.f_w), heads=self.num_classes)

        # from IPython import embed;embed()
        self.proj = ConvModule(
            self.channels,
            self.in_channels[self.size_index],
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            self.in_channels[self.size_index] + self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * len(self.in_channels),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        outs = []
        # from IPython import embed
        # embed()
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[self.in_index[self.size_index]].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))

        x = self.fusion_conv(torch.cat(outs, dim=1))

        identity = x

        out, attn = self.attn(x)
        align_corners = self.align_corners
        # from IPython import embed
        # embed()
        out = self.proj(out)
        out = resize(
            out,
            size=identity.shape[2:],
            mode='bilinear',
            align_corners=align_corners)

        out = self.bottleneck(torch.cat((identity, out), dim=1))
        out = self.cls_seg(out)

        attn = attn.permute(1, 0, 2, 3)
        attn = torch.diagonal(attn, dim1=2, dim2=3).view(self.num_classes, -1, self.f_h, self.f_w)
        return out, attn




    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, att = self.forward(inputs)
        losses = self.losses([seg_logits, att], gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        out, maps = self.forward(inputs)
        return out
        # return out, maps.permute(1, 0, 2, 3)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        attn = seg_logit[1]
        attn = resize(
            input=attn,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logit = seg_logit[0]
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            # class_weight=self.class_weight,
            ignore_index=self.ignore_index)

        loss['loss_seg'] += self.attn_loss(
            attn,
            seg_label,
            weight=seg_weight,
            # class_weight=self.class_weight,
            ignore_index=self.ignore_index)



        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        _, pred_label = seg_logit.topk(1, dim=1)
        b, _, h, w = seg_logit.size()
        pred_label = pred_label.view(b, h, w)

        if self.update_eval and not self.loss_decode.static_weight:
            acc = eval_metrics(pred_label.detach().cpu(), seg_label.detach().cpu(), self.num_classes, -1)
            class_acc = acc[1]
            class_acc[class_acc != class_acc] = 0
            total = torch.bincount(seg_label.view(-1), minlength=self.num_classes)[:self.num_classes].to(self.total_count.device)
            self.correct_count += torch.Tensor([x[0] * x[1] for x in zip(total, class_acc)])
            self.total_count += total

        return loss

