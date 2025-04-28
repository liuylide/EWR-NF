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
from utils.MAL import MAL_Astigmatism,MAL,defocusfitting
from utils.CTF_yy import CTF, makeFourierCoords
from SSIM import SSIM
#选择数据
from data.I20240807 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 3407
seed = 3270
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)  
DEBUG = False

def train():
    #读取文件
    if type_dm == True:
        image, dimensions, calibration, metadata = dm.dm_load(path)
        image = image.transpose([1,2,0])
    else:
        image1 = io.loadmat(path)
        image = np.array(image1[path1],dtype='float32')
    images = torch.Tensor(image)
    resolution_x,resolution_y,active = image.shape
    
    global defocus_series,i_train
    x = focalSeries(images,pixelsize,defocus_series)
    images1,images2 = x.align(align_x,align_y)


    resolution_x = 1000
    resolution_y = 1000
    offset_x = torch.arange(resolution_x+padsize).view(-1, 1).repeat(1, resolution_x+padsize)
    offset_y = torch.arange(resolution_y+padsize).repeat(resolution_y+padsize, 1)
    offset = torch.stack([offset_x, offset_y], dim=-1)[None]
    inputch = 4 
    level = 32
    mlr_mod = MultilevelResolutionInterpolation(inputch, level)
    mlr_mod = torch.nn.DataParallel(mlr_mod)
    mlr_mod = mlr_mod.to(device)
    optimizer_mlr_mod = torch.optim.Adam(mlr_mod.parameters(), 
                                    lr=5e-4, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-8, 
                                    weight_decay=0)
                                    
    model_sin = mlpNet(D=3,H=64,input_ch= inputch * level)
    model_sin = torch.nn.DataParallel(model_sin)
    model_sin = model_sin.to(device)
    optimizer_sin = torch.optim.Adam(model_sin.parameters(), 
                                    lr=5e-4, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-8, 
                                    weight_decay=0)

    model_cos = mlpNet(D=3,H=64,input_ch= inputch * level)
    model_cos = torch.nn.DataParallel(model_cos)
    model_cos = model_cos.to(device)
    optimizer_cos = torch.optim.Adam(model_cos.parameters(), 
                                    lr=5e-4, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-8, 
                                    weight_decay=0)
    
    #参数设置
    imx = images1
    result_name = '/public/home/liuyang2022/TEM/result/im1.mat'
    io.savemat(result_name, {'im1':imx.cpu().numpy()})
    criterion = SSIM()
    MSE=[]
    print('Begin')
    start = 0
    time0 = time.time()

    defocus_series = defocus_series[i_train]
    images1 = images1[:,:,i_train].float()
    images2 = images2[:,:,i_train].float()
    if imagetype == 'simulated':
        images1 = images2


    matrix =  makeFourierCoords(resolution_x + padsize,resolution_y + padsize,pixelsize)
    #开始训练
    for i in trange(N_iters):
        

        selected_image = np.random.choice(len(i_train))
        k = int(selected_image)

        grid = torch.rand(1, resolution_x+padsize, resolution_y+padsize, 2)
        grid_input = (offset+grid) / ((resolution_x+padsize) / 2) - 1 
        input_mod = mlr_mod(grid_input)

        network_sin = model_sin(input_mod)
        network_res_real = network_sin.reshape([resolution_x + padsize, resolution_y + padsize])

        network_cos = model_cos(input_mod)
        network_res_imag = network_cos.reshape([resolution_x + padsize, resolution_y + padsize])
        
        img1 = network_res_real  + 1j * network_res_imag 
        defocus = defocus_series[k].clone().detach().requires_grad_(True)



        if ifAstigmatism == True:
            img = MAL_Astigmatism(defocus,resolution_x + padsize,resolution_y + padsize,pixelsize,img1,
                E0 = E0,Cs = Cs,alpha = alpha,M = 5,deltaC1 = deltaC1,theta=0,z=150)
        else:
            img = MAL(defocus,resolution_x + padsize,resolution_y + padsize,pixelsize,img1,
                E0 = E0,Cs = Cs,alpha = alpha,M = 5,deltaC1 = deltaC1)


        if i <= N_iters *2/3 and ifdefoucsfitting != False:
            optimizer_defocus = torch.optim.Adam([defocus], 
                                lr=5e-1, 
                                betas=(0.9, 0.999), 
                                eps=1e-8, 
                                weight_decay=0)
            optimizer_defocus.zero_grad()
        if i <= N_iters *4/5:
            optimizer_sin.zero_grad()
            optimizer_mlr_mod.zero_grad()
        optimizer_cos.zero_grad()



        #比较传播出的图像与处理后的实拍图像的差距
        img = img.unsqueeze(0).unsqueeze(0)
        image1 = images1[44:1044,44:1044,k]
        image1 = image1.unsqueeze(0).unsqueeze(0)
        loss = 1 - criterion(img[:,:,44:1044,44:1044], image1)
        MSE.append(float(loss))
        loss.backward()

        #更新神经网络中的权重      


        if i <= N_iters *2/3 and ifdefoucsfitting != False:
            optimizer_defocus.step()
        if i <= N_iters *4/5:
            optimizer_sin.step()
            optimizer_mlr_mod.step()
        optimizer_cos.step() 

        defocus_series[k] = defocus
        decay_steps = N_iters

        initial_lr = 4e-4
        decay_rate1 = 1e-3
        new_lrate1 = initial_lr * (decay_rate1 ** (i / decay_steps))

        opt_initial_lr = 5e-4
        opt_decay_rate = 1e-2
        new_lrate = opt_initial_lr * (opt_decay_rate ** (i / decay_steps))


        for param_group in optimizer_sin.param_groups:
            param_group['lr'] = new_lrate1
        for param_group in optimizer_cos.param_groups:
            param_group['lr'] = new_lrate1
        for param_group in optimizer_mlr_mod.param_groups:
            param_group['lr'] = new_lrate

        #保存结果
        
        if (i + 1) % 20000 == 0:

            result_name = '/public/home/liuyang2022/TEM/result/MINIREALepoch' + diffstr + str(i+1) +'.mat'
            io.savemat(result_name, {'real': network_res_real.cpu().data.numpy().squeeze()})
            result_name = '/public/home/liuyang2022/TEM/result/MINIimagepoch' + diffstr + str(i+1) +'.mat'
            io.savemat(result_name, {'imag': network_res_imag.cpu().data.numpy().squeeze()})
            model_mlr_name = '/public/home/liuyang2022/TEM/model/Mlr' + diffstr + str(i+1) +'.pt'
            torch.save(mlr_mod, model_mlr_name)
            model_sin_name = '/public/home/liuyang2022/TEM/model/Sin' + diffstr + str(i+1) +'.pt'
            torch.save(model_sin, model_sin_name)
            model_cos_name = '/public/home/liuyang2022/TEM/model/Cos' + diffstr + str(i+1) +'.pt'
            torch.save(model_cos, model_cos_name)
            dt = time.time() - time0
            print(i,'/',N_iters,'iter  ', 'time: ', dt, 'loss: ', loss)
            time0 = time.time()
    result_name = '/public/home/liuyang2022/TEM/result/MINIREALepoch' + diffstr + str(i+1) +'.mat'
    io.savemat(result_name, {'real': network_res_real.cpu().data.numpy().squeeze()})
    result_name = '/public/home/liuyang2022/TEM/result/MINIimagepoch' + diffstr + str(i+1) +'.mat'
    io.savemat(result_name, {'imag': network_res_imag.cpu().data.numpy().squeeze()})
    result_name = '/public/home/liuyang2022/TEM/result/FS4epoch' + diffstr + str(N_iters) +'MSE.mat'
    io.savemat(result_name, {'loss':MSE})    
    
    return
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()