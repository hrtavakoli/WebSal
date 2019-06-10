'''
Loss function often used in saliency models

@author: Hamed R. Tavakoli
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_data(x):
    x = x.view(x.size(0), -1)
    x_sum = torch.sum(x, dim=1)
    x = x / x_sum.unsqueeze(1).expand_as(x)
    return x


def reshape_data(x):
    x = x.view(x.size(0), -1)
    x_max, idx = torch.max(x, dim=1, keepdim=True)
    x = x / (x_max.expand_as(x) + 1e-8)
    return x


class KLLoss(nn.Module):

    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, x0, y0):
        x0 = normalize_data(x0)
        y0 = normalize_data(y0)
        x_log = torch.log(x0 + 1e-8)
        y_log = torch.log(y0 + 1e-8)
        loss = torch.sum(torch.mul(y0, (y_log - x_log)), dim=1)

        return loss.sum()


class ACCLoss(nn.Module):

    def __init__(self):
        super(ACCLoss, self).__init__()

    def forward(self, x0, y0):

        x0 = normalize_data(x0)
        y0 = normalize_data(y0)

        msx = x0 - x0.mean(dim=1, keepdim=True).expand_as(x0)
        msy = y0 - y0.mean(dim=1, keepdim=True).expand_as(y0)

        denom = torch.sqrt(torch.sum(torch.pow(msx, 2), dim=1)) * torch.sqrt(torch.sum(torch.pow(msy, 2), dim=1))

        # we opimize 1 - rho which means our cost function is at the range of [0-2] where 2 is total disagreement
        loss = 1 - ( torch.sum(msx * msy, dim=1) / (denom + 1e-8) )

        return loss.sum()


class NEGNSSLoss(nn.Module):

    def __init__(self):
        super(NEGNSSLoss, self).__init__()

    def forward(self, pred, gt_fix):

        pred = reshape_data(pred)
        gt_fix[gt_fix > 0.1] = 1.0
        gt_fix = gt_fix.view(gt_fix.size(0), -1)

        p_mean = pred.mean(dim=1, keepdim=True).expand_as(pred)
        p_std = pred.std(dim=1, keepdim=True).expand_as(pred)
        pred = (pred - p_mean) / (p_std + 1e-8)

        nss = torch.sum(gt_fix * pred, dim=1, keepdim=True) / gt_fix.sum(dim=1, keepdim=True)
        nss_e = torch.exp(-nss)
        return nss_e.sum()
