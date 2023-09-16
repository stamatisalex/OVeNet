# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import wandb
import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank
from utils.coloring import colorize_predictions,original_palette,illustrate_pred, offset2flow
from torchvision.utils import make_grid
from PIL import Image

from .flow_vis import flow_to_color, flow_uv_to_colors, make_colorwheel

# seed = 3
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device,vis=False,sv_dir=''):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, labels, instances,name,_ = batch # ua xreiastei to instances

        images = images.to(device)
        labels = labels.long().to(device)
        # With o_f
        # losses, _,o_f  = model(images, labels) #inputs, labels

        # Without o_f
        losses, s_f, preds = model(images, labels)  # inputs, labels
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)



        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), lr, print_loss)
            logging.info(msg)

            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, model, writer_dict, device):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ ,name = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            # With o_f
            losses, pred,_ = model(image, label)
            # Without o_f
            # losses, pred = model(image, label)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            # for n, p in model.named_parameters():
            #     if p.grad is None:
            #         print(f'{n} has no grad')

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array
    

def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False,vis=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            if(vis):
                image, label, _, name,_,color = batch
            else:
                image, label, _, name,_ = batch
            size = label.size()
            names = name[0][:-16]
            # print(1)
            if(vis):
                pred,scores_pred,s_s_pred = test_dataset.multi_scale_inference(
                            model,
                            image,
                            scales=config.TEST.SCALE_LIST,
                            flip=config.TEST.FLIP_TEST,
                            debug=vis)
            else:
                pred = test_dataset.multi_scale_inference(
                            model,
                            image,
                            scales=config.TEST.SCALE_LIST,
                            flip=config.TEST.FLIP_TEST,
                            debug=False)
            # print(6)
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')
            if(vis):
                preds = model(image)
                offset_pred = preds[1]
                f = preds[4]
                # f2 = preds[5]
                # offset = preds[5]
                if offset_pred.size()[-2] != size[-2] or offset_pred.size()[-1] != size[-1]:
                    offset_pred = F.upsample(offset_pred, (size[-2], size[-1]),
                                      mode='bilinear')
                if f.size()[-2] != size[-2] or f.size()[-1] != size[-1]:
                    f = F.upsample(f, (size[-2], size[-1]),
                                      mode='bilinear')
                # if offset.size()[-2] != size[-2] or offset.size()[-1] != size[-1]:
                #     offset = F.upsample(offset, (size[-2], size[-1]),
                #                       mode='bilinear')
                # if f2.size()[-2] != size[-2] or f2.size()[-1] != size[-1]:
                #     f2 = F.upsample(f2, (size[-2], size[-1]),
                #                    mode='bilinear')
                if scores_pred.size()[-2] != size[-2] or scores_pred.size()[-1] != size[-1]:
                    scores_pred = F.upsample(scores_pred, (size[-2], size[-1]),
                                      mode='bilinear')
                if s_s_pred.size()[-2] != size[-2] or s_s_pred.size()[-1] != size[-1]:
                    s_s_pred = F.upsample(s_s_pred, (size[-2], size[-1]),
                                      mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if (vis):


                scores_pred = test_dataset.illustrate_pred(scores_pred)
                s_s_pred = test_dataset.illustrate_pred(s_s_pred)
                pred= test_dataset.illustrate_pred(pred)

                # Confidence map
                f=colorize_predictions(f,cmap="Spectral")
                # f_map = np.array(f, dtype=np.uint8)
                f_map = Image.fromarray(f)


                # in case of both offset and non offset confidence
                # f2 = colorize_predictions(f2,cmap="Spectral")
                # f_map2 = Image.fromarray(f2)

                sv_path = os.path.join(sv_dir, 'offset_validation_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)

                # offset = offset.cpu().detach().numpy()
                # flow_color_2 = flow_to_color(np.moveaxis(offset[0],0,-1), convert_to_bgr=False)


                # first way

                offset_pred = offset_pred.cpu().detach().numpy()
                flow_color = flow_to_color(np.moveaxis(offset_pred[0], 0, -1), convert_to_bgr=False)

                # second way
                # offset = offset_pred.permute(0,2,3,1)
                # u = offset[0, :, :, 0].cpu().numpy()
                # v = offset[0, :, :, 1].cpu().numpy()
                # img_flow_np = offset2flow(u, v)
                # flow_color = torch.from_numpy(img_flow_np.transpose((2, 0, 1)))/ 255.0



                mask ={
                    0:'road',
                    1:'sidewalk',
                    2:'building',
                    3:'wall',
                    4:'fence',
                    5:'polegroup',
                    6:'traffic light',
                    7:'traffic sign',
                    8:'vegetation',
                    9:'terrain',
                    10:'sky',
                    11:'person',
                    12:'rider',
                    13:'car',
                    14:'truck',
                    15:'bus',
                    16:'train',
                    17:'motorcycle',
                    18:'bicycle'
                }

                    # Log the images as wandb Image
                wandb.log({
                    "RGB": [wandb.Image(image[0], caption=f"Images " + names)],
                    "Semantic GT": [
                        wandb.Image(color[0], caption=f"Semantic GT " +  names)],
                    "Si ": [
                        wandb.Image(scores_pred[0], caption=f" S_i_" +  names)],
                    "Ss ": [wandb.Image(s_s_pred[0],
                                                 caption=f" S_s_ " + names)],
                    "Sf ": [
                        wandb.Image(pred[0], caption=f" S_f_"  + names)],
                    "Confidence Map": [
                        wandb.Image(f_map, caption=f"Confidence Map  " + names)],
                    # "Confidence Map Without Offsets": [
                    #     wandb.Image(f_map2, caption=f"Confidence Map  Without Offset" + names)],
                    "Offsets": [
                        wandb.Image(Image.fromarray(flow_color), caption=f"Offsets " + names)]
                    # "Offsets": [
                    #     wandb.Image(make_grid(flow_color), caption=f"Offsets " + names)]
                })

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
