# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.functions_plane import *



dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor
# device = torch.device('cuda:{}'.format(args.local_rank))
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target, flag=True):
        #score-> prediction
        #target->label
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, s_i, target,flag=True, **kwargs):
        ph, pw = s_i.size(2), s_i.size(3)
        h, w = target.size(1), target.size(2) #ground truth
        if ph != h or pw != w:
            s_i = F.upsample(input=s_i, size=(h, w), mode='bilinear')
        if(flag):
            pred = F.softmax(s_i, dim=1)
        else:
            pred = s_i
        pixel_losses = self.criterion(s_i, target).contiguous().view(-1) # batch * 512 * 1024 ~ 100000
        mask = target.contiguous().view(-1) != self.ignore_label   #The above line creates a Binary tensor that has a False at each place for the value=ignore_index
        
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





class Confidence_Loss(nn.Module):
    def __init__(self, device, ignore_label=-1):
        super().__init__()
        self.ignore_label = ignore_label
        self.device = device

    def calculate_H_seed(self,target,x,y):

        # floor function in order to do nearest neighbor algorithm
        x = torch.floor(x).type(dtype_long)
        y = torch.floor(y).type(dtype_long)

        x = torch.clamp(x, 0, target.shape[2] - 1)
        y = torch.clamp(y, 0, target.shape[1] - 1)

        return target[:,y,x]


    def forward(self,o_f,target, w_f=1, **kwargs):
        batch_size,ph, pw = o_f.size(0), o_f.size(2), o_f.size(3) #batch size to check
        h, w = target.size(1), target.size(2)  # h->512 , w->1024

        # coordinate map
        # w=1024
        # h=512
        # x -> [0, ... ,w] , y-> [0, ... , h]

        xm = torch.linspace(0, w-1, w).view(
            1, 1, -1).expand(1, h, w)     # 1 x h x w
        ym = torch.linspace(0, h-1, h).view(
            1, -1, 1).expand(1, h, w)
        xym = torch.cat((xm, ym), 0)          # 1 x h x w
        xym = xym.to(self.device)
        if ph != h or pw != w:
            o_f = F.upsample(input=o_f, size=(h, w), mode='bilinear')

        xym_s = xym[:, 0:h, 0:w].contiguous()  # 2 x h x w
        tmp_target = target.clone() # batch x h x w
        tmp_target[tmp_target == self.ignore_label] = 0 #ground truth

        f_loss = 0
        eps = 1e-7
        for b in range(0,batch_size):
            f = o_f[b, 2]  # h x w
            spatial_pix = o_f[b, 0:2] + xym_s  # 2 x h x w
            #Scaling

            x_cords = spatial_pix[0]  # h x w
            y_cords = spatial_pix[1]  # h x w

            #Target map ofsset vectors prediction
            H_s = self.calculate_H_seed(tmp_target[b].unsqueeze(0),x_cords,y_cords) # 1 x h x w

            mask = tmp_target[b] == H_s.squeeze(0) # h x w
            mask2 = mask < 1 # logical not
            f_loss+= (torch.sum(-torch.log(f[mask]+eps)) + torch.sum(-torch.log(1-f[mask2]+eps))) / (h*w)

        f_loss=f_loss/(b+1)
        loss = w_f * f_loss

        return loss

class Confidence_Loss_2(nn.Module):
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
        # f_loss = (torch.sum(-torch.log(f[mask ivate.unsqueeze(1)] + eps)) + torch.sum(-torch.log(1 - f[mask2 & mask_initial.unsqueeze(1)] + eps))) / (h * w)
        f_loss = (torch.sum(-torch.log(f[mask] + eps)) + torch.sum(-torch.log(1 - f[mask2 ] + eps))) / (h * w)

        return f_loss.mean()


class Size_Loss(nn.Module):
    def __init__(self,treshold=0.1):
        super().__init__()
        self.treshold = torch.tensor(treshold)

    def forward(self,offset, **kwargs):
        batch,_,h,w=offset.size()
        offset = offset.reshape(batch,2,h*w)
        offset = torch.norm(offset,dim=1)
        maximum =torch.maximum((torch.norm(offset,dim=1) - self.treshold),torch.tensor(0))
        return torch.pow(maximum,2)
