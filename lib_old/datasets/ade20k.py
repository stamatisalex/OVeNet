# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
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
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(ADE20K, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {
                'img': image_path,
                'label': label_path,
                'name': name
            }
            files.append(sample)
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
            image, label = self.rand_crop(image, label)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

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
            # print("preds",preds.size())
            # print("final_pred",final_pred.size())
            final_pred += preds
            if (debug):
                s_s_preds = F.upsample(s_s_preds, (ori_height, ori_width),
                                       mode='bilinear')
                scores_preds = F.upsample(scores_preds, (ori_height, ori_width),
                                          mode='bilinear')

                # print("a",offset_final_pred.size())
                # print("b",offset_preds.size())
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
        palette = [128, 64, 128,
                   244, 35, 232,
                   70, 70, 70,
                   102, 102, 156,
                   190, 153, 153,
                   153, 153, 153,
                   250, 170, 30,
                   220, 220, 0,
                   107, 142, 35,
                   152, 251, 152,
                   70, 130, 180,
                   220, 20, 60,
                   255, 0, 0,
                   0, 0, 142,
                   0, 0, 70,
                   0, 60, 100,
                   0, 80, 100,
                   0, 0, 230,
                   119, 11, 32]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        return palette

    def illustrate_pred(self, preds):
        # palette = self.get_palette(256)
        palette = self.original_palette()
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        preds_list = []
        for i in range(preds.shape[0]):
            # print("huston",preds[i])
            # pred=self.convert_label(preds[i], inverse=False)
            # print(pred)
            save_img = Image.fromarray(preds[i])
            save_img.putpalette(palette)

            preds_list.append(save_img)
        return preds_list

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            # pred = self.convert_label(preds[i], inverse=True)
            pred = preds[i]
            # print(pred)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))



