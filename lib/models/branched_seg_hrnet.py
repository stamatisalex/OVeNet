# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from models.functions_plane import *



BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True,branch=False):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)

        self.branch = branch
        self.fuse_layers = self._make_fuse_layers(branch)
        # self.fuse_layers_branched = self._make_fuse_layers()

        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self,branch):

        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        if self.branch:
            fuse_layers2 = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            if (self.branch):
                fuse_layer2 = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    if (self.branch):
                        fuse_layer2.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[j],
                                      num_inchannels[i],
                                      1,
                                      1,
                                      0,
                                      bias=False),
                            BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                    if (self.branch):
                        fuse_layer2.append(None)
                else:
                    conv3x3s = []
                    if (self.branch):
                        conv3x3s2 = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                            if (self.branch):
                                conv3x3s2.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j],
                                              num_outchannels_conv3x3,
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3,
                                                momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                            if(self.branch):
                                conv3x3s2.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j],
                                              num_outchannels_conv3x3,
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3,
                                                momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))  # etsi ua kano to fusion sta seeds
                    if(self.branch):
                        fuse_layer2.append(nn.Sequential(*conv3x3s2))
            fuse_layers.append(nn.ModuleList(fuse_layer))
            if (self.branch):
                fuse_layers2.append(nn.ModuleList(fuse_layer2))
        if(self.branch):
            return nn.ModuleList([nn.ModuleList(fuse_layers) ,nn.ModuleList(fuse_layers2)]) #thelei fuse_layers2
        else:
            return nn.ModuleList(fuse_layers)



    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]


        for i in range(self.num_branches):
                x[i] = self.branches[i](x[i])

        x_fuse = []

        if (self.branch):

            x2_fuse = [] #for the extra layer
            for i in range(len(self.fuse_layers[0])):  # i ranges from 0 to 7
                y = x[0] if i == 0 else self.fuse_layers[0][i][0](x[0])
                y1= x[0] if i == 0 else self.fuse_layers[1][i][0](x[0])
                for j in range(1, self.num_branches):  # j ranges from 0 to 3
                    if i == j:
                        y = y + x[j]
                        y1 = y1 + x[j]
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        # print(type(t1[i][j](x[j])))
                        y = y + F.interpolate(
                            self.fuse_layers[0][i][j](x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                        y1 = y1 + F.interpolate(
                            self.fuse_layers[1][i][j](x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                    else:
                        y = y + self.fuse_layers[0][i][j](x[j])
                        y1 = y1 + self.fuse_layers[1][i][j](x[j])
                x_fuse.append(self.relu(y))
                x2_fuse.append(self.relu(y1))
            x_fuse=x_fuse+x2_fuse
        else:
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        y = y + F.interpolate(
                            self.fuse_layers[i][j](x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
                x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

#NEW IMPLEMENTATION OF THE NETWORK

class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()
        self.ex=extra
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.offset_branch = int(extra.OFFSET_BRANCH)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        if (self.offset_branch == 2):
            self.transition1_2 = self._make_transition_layer(
                [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)
        if (self.offset_branch == 2):
            self.stage2_2_cfg = extra['STAGE2_2']
            self.stage2_2, pre_stage_channels = self._make_stage(
                self.stage2_2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        if(self.offset_branch <= 3 ):
            self.transition2_2 = self._make_transition_layer(
                pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        if(self.offset_branch <= 3):
            self.stage3_2_cfg = extra['STAGE3_2']
            self.stage3_2, pre_stage_channels=self._make_stage(
                self.stage3_2_cfg, num_channels)



        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        if (self.offset_branch<= 4 ):
            self.transition3_2 = self._make_transition_layer(
                pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, branch=False)
        if (self.offset_branch<= 4):
            self.stage4_2_cfg = extra['STAGE4_2']
            self.stage4_2, pre_stage_channels = self._make_stage(
                self.stage4_2_cfg, num_channels, multi_scale_output=True, branch=False)

        last_inp_channels = np.int(np.sum(pre_stage_channels))
        # print('last inp channels',last_inp_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

        self.offset_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=3,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_SEED == 3 else 0)
        )
        # coordinate map

        # x ->  [ 0 ... 1023]
        #            .
        #            .
        #       [ 0 ... 1023]

        # y ->  [ 0 ... 0]
        #            .
        #            .
        #      [511 ... 511]
        self.refinement = int(extra.ITERATIVE_REFINEMENT)
        self.offset_threshold = extra.OFFSET_THRESHOLD
        self.get_coords = get_coords
        self.batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
        self.H, self.W = config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]
        coords = self.get_coords(self.batch_size, self.H, self.W, fix_axis=True)
        self.coords = nn.Parameter(coords, requires_grad=False)
        self.freeze=config.MODEL.FREEZED_PAR
        self.logits = extra.LOGITS
        self.confidence = int(extra.CONFIDENCE)


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        print("flexxx", num_branches_cur)
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, branch=False):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output, branch)
            )
            # else:
            #     modules.append(
            #         HighResolutionModule2(num_branches,
            #                              block,
            #                              num_blocks,
            #                              num_inchannels,
            #                              num_channels,
            #                              fuse_method,
            #                              reset_multi_scale_output)
            #     )
            num_inchannels = modules[-1].get_num_inchannels()
            print("hey_channels", num_inchannels)
        # print('num_inchannels',num_inchannels)
        # print('lenght is',len(modules))
        # if(branch):
        #     print(nn.Sequential(*modules))
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        if(self.offset_branch == 2):
            x2_list = []
            for i in range(self.stage2_cfg['NUM_BRANCHES']):
                if self.transition1_2[i] is not None:
                    x2_list.append(self.transition1_2[i](x))
                else:
                    x2_list.append(x)
            y2_list = self.stage2_2(x2_list)

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        if(self.offset_branch <= 3):
            x2_list = []
            if(self.offset_branch <3):
                y_list = y2_list
            else:
                y_list = y_list
            for i in range(self.stage3_cfg['NUM_BRANCHES']):
                if self.transition2_2[i] is not None:
                    x2_list.append(self.transition2_2[i](y_list[-1]))
                else:
                    x2_list.append(y_list[i])
            y2_list = self.stage3_2(x2_list)

        y_list = self.stage3(x_list)
        # print('print layer 3', y_list)
        x_list = []

        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        if (self.offset_branch <= 4):
            x2_list = []
            if(self.offset_branch<4):
                y_list = y2_list
            else:
                y_list = y_list
            for i in range(self.stage4_cfg['NUM_BRANCHES']):
                if self.transition3_2[i] is not None:
                    x2_list.append(self.transition3_2[i](y_list[-1]))
                else:
                    x2_list.append(y_list[i])

        x = self.stage4(x_list)
        x_2 = self.stage4_2(x2_list)
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        # Extra layer
        x2_h, x2_w = x_2[0].size(2), x_2[0].size(3)
        x2_1 = F.upsample(x_2[1], size=(x2_h, x2_w), mode='bilinear')
        x2_2 = F.upsample(x_2[2], size=(x2_h, x2_w), mode='bilinear')
        x2_3 = F.upsample(x_2[3], size=(x2_h, x2_w), mode='bilinear')

        x_1 = torch.cat([x[0], x1, x2, x3], 1)  # batch x 720 x 128 x 256
        x_2 = torch.cat([x_2[0], x2_1, x2_2, x2_3], 1)
        # x = torch.cat([x[0],x[4],x1,x5,x2,x6,x3,x7],1)
        # h=128 , w=256 whereas the inital dimensions were 512 x 1024

        # initial prediciton of the network at p
        scores = self.last_layer(x_1)  # batch x 19 x h x w
        batch_size,C,H,W = scores.size()
        if ( not self.logits):
            scores = F.softmax(scores, dim=1)
        if self.H != H or self.W != W:
            coords = self.get_coords(batch_size, H, W, fix_axis=True) # batch x h x w x 2
            ocoords_orig = nn.Parameter(coords, requires_grad=False)
        else:
            ocoords_orig = self.coords
            if self.batch_size > batch_size:
                ocoords_orig = self.coords[0:batch_size]

        o_f =self.offset_layer(x_2) # batch x 3 x h x w
        f = self.sigmoid(o_f[:,2]).unsqueeze(1) # batch x 1 x h x w
        offset = self.tanh(o_f[:,0:2]) * float(self.offset_threshold)
        offset = offset.permute(0, 2, 3, 1) # batch x h x w x 2
        ocoords = ocoords_orig + offset # batch x h x w x 2
        ocoords = torch.clamp(ocoords, min=-1.0, max=1.0)

        f_offset = f

        if self.refinement > 0:
            # offset_ref = offset
            for _ in range(0, self.refinement):
                du = offset[:, :, :, 0].unsqueeze(1)
                dv = offset[:, :, :, 1].unsqueeze(1)
                du = du + F.grid_sample(du, ocoords, padding_mode="zeros")
                dv = dv + F.grid_sample(dv, ocoords, padding_mode="zeros")
                # f_offset = F.grid_sample(f_offset, ocoords, padding_mode="zeros")
                offset = torch.cat([du, dv], dim=1)
                # offset_ref = offset_ref.permute(0,3,1,2) # batch x 2 x h x w
                # offset_ref = offset_ref + F.grid_sample(offset_ref, ocoords, padding_mode="zeros")
                # offset_ref = offset_ref.permute(0, 2, 3, 1) # batch x h x w x 2
                # ocoords = ocoords_orig + offset_ref
                # ocoords = torch.clamp(ocoords, min=-1.0, max=1.0)
                offset = offset.permute(0, 2, 3, 1) # batch x h x w x 2
                ocoords = ocoords_orig + offset
                ocoords = torch.clamp(ocoords, min=-1.0, max=1.0)

        s_s = F.grid_sample(scores, ocoords, padding_mode="border")


        # f_offset = F.grid_sample(f_offset, ocoords, padding_mode="zeros")

        if (self.confidence) == 0:
            confidence_map = f
        elif (self.confidence) == 1:
            confidence_map = f_offset
        else:
            raise ValueError('The specified confidence is not implemented')

        s_f = (1-confidence_map)*scores + confidence_map * s_s
        offset = offset.permute(0, 3, 1, 2)

        return scores, offset , s_s, s_f, confidence_map


    def init_weights(self, pretrained='',pretrained_offset=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            print(m)


            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            pretrained_dict_offset = torch.load(pretrained_offset)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            logger.info('=> loading pretrained offset model {}'.format(pretrained_offset))
            model_dict = self.state_dict()

            #Modify the pretrained_dict_offset . In fact we delete the required_grad = Fla



            #Modify the pretrained_dictionary for cityscapes
            #
            # pretrained_dict = {(k.replace('fuse_layers.', 'fuse_layers.0.') if k[6:].startswith('stage4') else k): v for k, v in
            #           pretrained_dict.items()}

            # Modify the pretrained_dictionary for imagenet

            # pretrained_dict = {(k.replace('fuse_layers.', 'fuse_layers.0.') if k.startswith('stage4') else k): v for k, v in
            #           pretrained_dict.items()}


            # For cityscapes weights
            if (self.freeze):
                if (self.offset_branch == 2):
                    pretrained_dict_offset = {
                        (k.replace('stage2.', 'stage2_2.') if k[6:].startswith('stage2') else k.replace(
                            'transition1.', 'transition1_2.') if k[6:].startswith('transition1') else
                        k.replace('stage3.', 'stage3_2.') if k[6:].startswith('stage3') else k.replace(
                            'transition2.', 'transition2_2.') if k[6:].startswith('transition2') else
                        k.replace('stage4.', 'stage4_2.') if k[6:].startswith('stage4') else k.replace(
                            'transition3.', 'transition3_2.') if k[6:].startswith('transition3') else ''): v for k, v in
                        pretrained_dict_offset.items()}
                elif (self.offset_branch == 3):
                    pretrained_dict_offset = {
                        (k.replace('stage3.', 'stage3_2.') if k[6:].startswith('stage3') else k.replace(
                            'transition2.', 'transition2_2.') if k[6:].startswith('transition2') else
                        k.replace('stage4.', 'stage4_2.') if k[6:].startswith('stage4') else k.replace(
                            'transition3.', 'transition3_2.') if k[6:].startswith('transition3') else ''): v for k, v in
                        pretrained_dict_offset.items()}
                elif (self.offset_branch == 4):
                    pretrained_dict_offset = {
                        (k.replace('stage4.', 'stage4_2.') if k[6:].startswith('stage4') else k.replace(
                            'transition3.', 'transition3_2.') if k[6:].startswith('transition3') else ''): v for k, v in
                        pretrained_dict_offset.items()}

                pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                                   if k[6:] in model_dict.keys()}

                pretrained_dict_offset = {k[6:]: v for k, v in pretrained_dict_offset.items()
                                   if k[6:] in model_dict.keys()}
            # For imagenet weights
            else:
                if(self.offset_branch ==2):
                    pretrained_dict_offset = {(k.replace('stage2.', 'stage2_2.') if k.startswith('stage2') else k.replace(
                        'transition1.', 'transition1_2.') if k.startswith('transition1') else
                                                  k.replace('stage3.', 'stage3_2.') if k.startswith('stage3') else k.replace(
                        'transition2.', 'transition2_2.') if k.startswith('transition2') else
                                                  k.replace('stage4.', 'stage4_2.') if k.startswith('stage4') else k.replace(
                        'transition3.', 'transition3_2.') if k.startswith('transition3') else ''): v for k, v in
                                              pretrained_dict_offset.items()}
                elif(self.offset_branch ==3):
                    pretrained_dict_offset = {(k.replace('stage3.', 'stage3_2.') if k.startswith('stage3') else k.replace(
                        'transition2.', 'transition2_2.') if k.startswith('transition2') else
                                                  k.replace('stage4.', 'stage4_2.') if k.startswith('stage4') else k.replace(
                        'transition3.', 'transition3_2.') if k.startswith('transition3') else ''): v for k, v in
                                              pretrained_dict_offset.items()}
                elif (self.offset_branch == 4):
                    pretrained_dict_offset = {(k.replace('stage4.', 'stage4_2.') if k.startswith('stage4') else k.replace(
                        'transition3.', 'transition3_2.') if k.startswith('transition3') else ''): v for k, v in
                                              pretrained_dict_offset.items()}

                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict.keys()}

                pretrained_dict_offset = {k: v for k, v in pretrained_dict_offset.items()
                                   if k in model_dict.keys()}

            for k, _ in pretrained_dict.items():
               logger.info(
                   '=> loading {} pretrained model'.format(k))
            model_dict.update(pretrained_dict)


            for k, _ in pretrained_dict_offset.items():
               logger.info(
                   '=> loading {} offset pretrained model'.format(k))
            model_dict.update(pretrained_dict_offset)


            # Lets initialize some weights

            # w = self.offset_layer
            # nn.init.constant_(w[0].weight,0)
            # nn.init.constant_(w[1].weight, 0)
            # nn.init.constant_(w[3].weight, 0)
            #
            #
            # model_dict['offset_layer.0.weight']= w[0].weight
            # model_dict['offset_layer.1.weight'] = w[1].weight
            # model_dict['offset_layer.3.weight'] = w[3].weight
            # print(model_dict)
            self.load_state_dict(model_dict)

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    # model.init_weights( cfg.MODEL.PRETRAINED.INITIAL, cfg.MODEL.PRETRAINED.OFFSET)
    model.init_weights(cfg.MODEL.PRETRAINED, cfg.MODEL.OFFSET_PRETRAINED)
    #   Parameters with names and requires_grad values
    if (cfg.MODEL.FREEZED_PAR):
        for param in model.parameters():
            param.requires_grad = False
        if(cfg.MODEL.EXTRA.OFFSET_BRANCH <=4):
            for param in model.stage4_2.parameters():
                param.requires_grad = True
            for param in model.transition3_2.parameters():
                param.requires_grad = True
            if (cfg.MODEL.EXTRA.OFFSET_BRANCH <= 3):
                for param in model.stage3_2.parameters():
                    param.requires_grad = True
                for param in model.transition2_2.parameters():
                    param.requires_grad = True
                if (cfg.MODEL.EXTRA.OFFSET_BRANCH <= 2):
                    for param in model.stage2_2.parameters():
                        param.requires_grad = True
                    for param in model.transition1_2.parameters():
                        param.requires_grad = True
        for param in model.offset_layer.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name} --> True')
            else:
                print(f'{name} --> False')
    return model