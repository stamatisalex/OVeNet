# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Stamatis Alexandropoulos (stamatisalex7@gmail.com)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image

from .base_dataset import BaseDataset


class ADE20K(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=150,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(520, 520),
                 downsample_rate=1,
                 vis = False,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(ADE20K, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.vis = vis
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]


    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                if(self.vis):
                    image_path, label_path,color_path = item
                    name = os.path.splitext(os.path.basename(label_path))[0]
                    files.append({
                        "img": image_path,
                        "label": label_path,
                        "color": color_path,
                        "name": name
                    })
                else:
                    image_path, label_path = item
                    name = os.path.splitext(os.path.basename(label_path))[0]
                    files.append({
                        "img": image_path,
                        "label": label_path,
                        "name": name
                    })

        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, 'ade20k', item['img'])
        label_path = os.path.join(self.root, 'ade20k', item['label'])
        # image_path = os.path.join(self.root, item['img'])
        # label_path = os.path.join(self.root, item['label'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        label = np.array(
            Image.open(label_path).convert('P')
        )
        label = self.reduce_zero_label(label)
        size = label.shape

        if 'testval' in self.list_path:
            image = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if 'val' in self.list_path:
            image, label = self.resize_short_length(
                image,
                label=label,
                short_length=self.base_size,
                fit_stride=8
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            if self.vis:
                colored_path = os.path.join(self.root, 'ade20k', item['color'])
                color = cv2.imread(
                    colored_path,
                    cv2.IMREAD_COLOR
                )
                color, _ = self.resize_short_length(
                    color,
                    label=label,
                    short_length=self.base_size,
                    fit_stride=8
                )

                color = self.input_transform(color)
                color = color.transpose((2, 0, 1))
                return image.copy(), label.copy(), np.array(size), name, color.copy()
            else:
                return image.copy(), label.copy(), np.array(size), name

        image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name


    def multi_scale_inference(self, config, model, image, scales=[1], flip=False, debug=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        if (debug):
            s_s_final_pred = torch.zeros([1, self.num_classes,
                                          ori_height, ori_width]).cuda()
            scores_final_pred = torch.zeros([1, self.num_classes,
                                             ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                # print(2)
                if (debug):
                    preds, scores_preds, s_s_preds = self.inference(model, new_img, flip, debug=True)
                    scores_preds = scores_preds[:, :, 0:height, 0:width]
                    s_s_preds = s_s_preds[:, :, 0:height, 0:width]
                else:
                    preds = self.inference(config, model, new_img, flip)

                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                if (debug):
                    s_s_preds = torch.zeros([1, self.num_classes,
                                             new_h, new_w]).cuda()
                    scores_preds = torch.zeros([1, self.num_classes,
                                                new_h, new_w]).cuda()
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        if (debug):
                            pred, scores_pred, s_s_pred = self.inference(model, crop_img, flip, debug=True)
                            scores_preds[:, :, h0:h1, w0:w1] += scores_pred[:, :, 0:h1 - h0, 0:w1 - w0]
                            s_s_preds[:, :, h0:h1, w0:w1] += s_s_pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        else:
                            pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
                if (debug):
                    scores_preds = scores_preds / count
                    scores_preds = scores_preds[:, :, :height, :width]
                    s_s_preds = s_s_preds / count
                    s_s_preds = s_s_preds[:, :, :height, :width]

            preds = F.upsample(preds, (ori_height, ori_width),
                               mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            final_pred += preds
            if (debug):
                s_s_preds = F.upsample(s_s_preds, (ori_height, ori_width),
                                       mode='bilinear')
                scores_preds = F.upsample(scores_preds, (ori_height, ori_width),
                                          mode='bilinear')
                s_s_final_pred += s_s_preds
                scores_final_pred += scores_preds

        if (debug):
            return final_pred, scores_final_pred, s_s_final_pred
        else:
            return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette


    def original_palette(self):
        palette = [120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,184,255,0,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,230,230,230,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        return palette

    def illustrate_pred(self, preds):
        palette = self.original_palette()
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        preds_list = []
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.putpalette(palette)

            preds_list.append(save_img)
        return preds_list

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i]
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
