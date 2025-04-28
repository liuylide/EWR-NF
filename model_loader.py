import torch.fft as fft
from torch.fft import fft2,ifft2
from tqdm import tqdm, trange
from model.MLP import *
import torch
import numpy as np
import time
import scipy
from scipy import io
import utils.DM as dm 
from utils.multiCorr import multiCorr, makeFourierCoords1
from utils.preprocess import focalSeries
from utils.loss import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_mlr = '/public/home/liuyang2022/TEM/model/Mlr_0403_prvk_2048_40000.pt'
path_cos = '/public/home/liuyang2022/TEM/model/Cos_0403_prvk_2048_40000.pt'
path_sin = '/public/home/liuyang2022/TEM/model/Sin_0403_prvk_2048_40000.pt'

mlr_mod = torch.load(path_mlr)
mlr_mod = mlr_mod.to(device)
mlr_mod.eval()

model_cos = torch.load(path_cos)
model_cos = model_cos.to(device)
model_cos.eval()

model_sin = torch.load(path_sin)
model_sin = model_sin.to(device)
model_sin.eval()

resolution_x = 1024
resolution_y = 1024

padsize = 96

offset_x = torch.arange(start= 750, end= 890,step = 0.125).view(-1, 1).repeat(1, resolution_x+padsize)
offset_y = torch.arange(start= 530, end= 670,step = 0.125).repeat(resolution_y+padsize, 1)

offset = torch.stack([offset_x, offset_y], dim=-1)[None]

# grid = torch.rand(1, resolution_x+padsize, resolution_y+padsize, 2)
grid_input = (offset) / ((resolution_x+padsize) / 2) - 1 

input_mod = mlr_mod(grid_input)

network_sin = model_sin(input_mod)
network_res_real = network_sin.reshape([resolution_x + padsize, resolution_y + padsize])

network_cos = model_cos(input_mod)
network_res_imag = network_cos.reshape([resolution_x + padsize, resolution_y + padsize])

img1 = network_res_real  + 1j * network_res_imag 
img1 = img1.to(device)

# from data.I20231229 import *
# from MAL import MAL
# from CTF_yy import CTF, makeFourierCoords
# for i in [16,17,18]:
#     defocus = defocus_series[i].to(device)
#     img = MAL(defocus,resolution_x + padsize,resolution_y + padsize,pixelsize,img1,
#                 E0 = E0,Cs = Cs,alpha = alpha,M = 5,deltaC1 = deltaC1)
#     result_name = '/public/home/liuyang2022/TEM/result/1229image' + str(i) +'.mat'
#     io.savemat(result_name, {'img': img.cpu().data.numpy().squeeze()})


diffstr = 'load0216'
result_name = '/public/home/liuyang2022/TEM/result/MINIREALepoch' + diffstr +'.mat'
io.savemat(result_name, {'real': network_res_real.cpu().data.numpy().squeeze()})
result_name = '/public/home/liuyang2022/TEM/result/MINIimagepoch' + diffstr +'.mat'
io.savemat(result_name, {'imag': network_res_imag.cpu().data.numpy().squeeze()})
