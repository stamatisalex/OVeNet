import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6

def get_coords(batch_size, H, W, fix_axis=False):
    U_coord = torch.arange(start=0, end=W).unsqueeze(0).repeat(H, 1).float()
    V_coord = torch.arange(start=0, end=H).unsqueeze(1).repeat(1, W).float()
    if not fix_axis:
        U_coord = (U_coord - ((W - 1) / 2)) / max(W, H)
        V_coord = (V_coord - ((H - 1) / 2)) / max(W, H)
    coords = torch.stack([U_coord, V_coord], dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    coords = coords.permute(0, 2, 3, 1).cuda()
    coords[..., 0] /= W - 1
    coords[..., 1] /= H - 1
    coords = (coords - 0.5) * 2
    return coords