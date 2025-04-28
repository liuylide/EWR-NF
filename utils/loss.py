import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class MSE_and_GDL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSE_and_GDL, self).__init__()

    def forward(self, inputs, targets, lambda_mse, lambda_gdl):

        squared_error = (inputs - targets).pow(2)
        gradient_diff_i = (inputs.diff(axis=-1)-targets.diff(axis=-1)).pow(2)
        gradient_diff_j =  (inputs.diff(axis=-2)-targets.diff(axis=-2)).pow(2)
        loss = (lambda_mse*squared_error.sum() + lambda_gdl*gradient_diff_i.sum() + lambda_gdl*gradient_diff_j.sum())/inputs.numel()
        # print("GDL:%f" %((gradient_diff_i.sum() + gradient_diff_j.sum())/inputs.numel()))
        # print("MSE:%f" %((squared_error.sum())/inputs.numel()) )

        return loss


class MAE_and_GDL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSE_and_GDL, self).__init__()

    def forward(self, inputs, targets, lambda_mse, lambda_gdl):

        squared_error = torch.abs(inputs - targets)
        gradient_diff_i = (inputs.diff(axis=-1)-targets.diff(axis=-1)).pow(2)
        gradient_diff_j =  (inputs.diff(axis=-2)-targets.diff(axis=-2)).pow(2)
        loss = (lambda_mse*squared_error.sum() + lambda_gdl*gradient_diff_i.sum() + lambda_gdl*gradient_diff_j.sum())/inputs.numel()

        return loss

def _tensor_size(t):
    return t.size()[0] * t.size()[1]
def tv_loss(x):

    h_x = x.size()[0]
    w_x = x.size()[1]

    count_h = _tensor_size(x[ 1:, :])
    count_w = _tensor_size(x[ :, 1:])
    h_tv = torch.pow((x[ 1:, :] - x[ :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[ :, 1:] - x[ :, :w_x - 1]), 2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)

class TV_Loss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TV_Loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size=x.shape[0]
        return self.TVLoss_weight*tv_loss(x)/batch_size


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)