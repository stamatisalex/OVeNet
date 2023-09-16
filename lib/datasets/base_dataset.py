# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

class BaseDataset(data.Dataset):
    def __init__(self, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)
    
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image
    
    def label_transform(self, label):
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=padvalue)
        
        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))
        
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def center_crop(self, image, label):
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.))
        y = int(round((h - self.crop_size[0]) / 2.))
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label
    
    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        
        image = cv2.resize(image, (new_w, new_h), 
                           interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), 
                           interpolation = cv2.INTER_NEAREST)
        else:
            return image
        
        return image, label

    def multi_scale_aug(self, image, label=None, 
            rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def gen_sample(self, image, label,
            multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, 
                                                    rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image, 
                                             self.base_size,
                                             label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)
        
        image = image.transpose((2, 0, 1))
        
        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label, 
                               None, 
                               fx=self.downsample_rate,
                               fy=self.downsample_rate, 
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def inference(self, model, image, flip=False, debug=False):
        size = image.size()
        outputs= model(image)
        scores = outputs[0]
        s_s = outputs[2]
        pred = outputs[3]

        # offset prediction
        if (debug):
            scores = F.upsample(input=scores,
                                     size=(size[-2], size[-1]),
                                     mode='bilinear')
            s_s = F.upsample(input=s_s,
                                     size=(size[-2], size[-1]),
                                     mode='bilinear')

        pred = F.upsample(input=pred,
                          size=(size[-2], size[-1]),
                          mode='bilinear')

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flips = model(torch.from_numpy(flip_img.copy()))
            scores_flip_output = flips[0]
            s_s_flip_output = flips[2]
            flip_output= flips[3]

            if (debug):
                scores_flip_output = F.upsample(input=scores_flip_output,
                                               size=(size[-2], size[-1]),
                                               mode='bilinear')
                scores_flip_output_pred = scores_flip_output.cpu().numpy().copy()
                scores_flip_output_pred = torch.from_numpy(scores_flip_output_pred[:, :, :, ::-1].copy()).cuda()
                scores += scores_flip_output_pred
                scores = scores * 0.5


                s_s_flip_output = F.upsample(input=s_s_flip_output,
                                               size=(size[-2], size[-1]),
                                               mode='bilinear')
                s_s_flip_output_pred = s_s_flip_output.cpu().numpy().copy()
                s_s_flip_output_pred = torch.from_numpy(s_s_flip_output_pred[:, :, :, ::-1].copy()).cuda()
                s_s += s_s_flip_output_pred
                s_s = s_s * 0.5


            flip_output = F.upsample(input=flip_output,
                                     size=(size[-2], size[-1]),
                                     mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        if (debug):
            return pred.exp(), scores.exp(), s_s.exp()
        else:
            return pred.exp()


    # def inference(self, model, image, flip=False,offset=False):
    #     size = image.size()
    #     pred = model(image)
    #
    #
    #     #offset prediction
    #     if (offset):
    #         offset_pred = pred[1]
    #         offset_pred = F.upsample(input=offset_pred,
    #                             size=(size[-2], size[-1]),
    #                             mode='bilinear')
    #
    #
    #     pred = pred[3]
    #     pred = F.upsample(input=pred,
    #                         size=(size[-2], size[-1]),
    #                         mode='bilinear')
    #
    #     if flip:
    #         flip_img = image.numpy()[:,:,:,::-1]
    #         flip_output = model(torch.from_numpy(flip_img.copy()))
    #
    #         if(offset):
    #             offset_flip_count = flip_output[1]
    #             offset_flip_count = F.upsample(input=offset_flip_count,
    #                                      size=(size[-2], size[-1]),
    #                                      mode='bilinear')
    #             offset_flip_pred = offset_flip_count.cpu().numpy().copy()
    #             offset_flip_pred = torch.from_numpy(offset_flip_pred[:, :, :, ::-1].copy()).cuda()
    #             offset_pred += offset_flip_pred
    #             offset_pred = offset_pred * 0.5
    #         flip_output = flip_output[3]
    #         flip_output = F.upsample(input=flip_output,
    #                         size=(size[-2], size[-1]),
    #                         mode='bilinear')
    #         flip_pred = flip_output.cpu().numpy().copy()
    #         flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
    #         pred += flip_pred
    #         pred = pred * 0.5
    #     if(offset):
    #         return pred.exp(), offset_pred.exp()
    #     else:
    #         return pred.exp()

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        device = torch.device("cuda:%d" % model.device_ids[0])
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).to(device)
        padvalue = -1.0  * np.array(self.mean) / np.array(self.std)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width, 
                                    self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width, 
                                        self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).to(device)
                count = torch.zeros([1,1, new_h, new_w]).to(device)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img, 
                                                      h1-h0, 
                                                      w1-w0, 
                                                      self.crop_size, 
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)

                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred


