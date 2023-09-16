# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Stamatis Alexandropoulos (stamatisalex7@gmail.com)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config
from models.functions_plane import *

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        try:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
        except IndexError:
            print("Error")
            return torch.tensor(float('nan'))

        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])


class Confidence_Loss(nn.Module):
    def __init__(self,ignore_label=-1):
        super().__init__()
        self.ignore_label = ignore_label
        self.get_coords = get_coords

    def forward(self,offset,f,target, **kwargs):
        batch_size,ph, pw = f.size(0), f.size(2),f.size(3)  # batch size to check
        h, w = target.size(1), target.size(2)  # h->512 , w->1024

        if ph != h or pw != w:
            f = F.upsample(input=f, size=(h, w), mode='bilinear')
            offset = F.upsample(input=offset, size=(h, w), mode='bilinear')

        coords = self.get_coords(batch_size, h, w, fix_axis=True)
        ocoords_orig = nn.Parameter(coords, requires_grad=False)

        mask_initial = target != self.ignore_label # batch x h x w
        tmp_target = target.clone()  # batch x h x w
        tmp_target[tmp_target == self.ignore_label] = 0  # ground truth
        tmp_target = tmp_target.type(dtype)
        eps = 1e-7
        # f --> batch x 1 x h x w
        # offset --> batch x 2 x h x w
        offset = offset.permute(0, 2, 3 ,1) # batch x h x w x 2
        ocoords = ocoords_orig + offset # batch x h x w x 2

        H_s = F.grid_sample(tmp_target.unsqueeze(1), ocoords,mode='nearest', padding_mode='border') # batch x 1 x h x w
        mask = tmp_target.unsqueeze(1) == H_s
        mask2 = mask < 1  # logical not
        f_loss = (torch.sum(-torch.log(f[mask] + eps)) + torch.sum(-torch.log(1 - f[mask2 ] + eps))) / (h * w)

        return f_loss.mean()