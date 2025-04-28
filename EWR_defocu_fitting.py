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
from utils.MAL import MAL,defocusfitting
#选择数据
from data.Si import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 3407
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
    # image = image[:,50:950,:]
    images = torch.Tensor(image)
    resolution_x,resolution_y,active = image.shape
    
    #对图像进行预处理
    global defocus_series,i_train
    # x = focalSeries(images,pixelsize,defocus_series1 = defocus_series,E0 = E0,padsize = padsize,Cs = Cs,alpha = alpha , deltaC1= deltaC1)
    images1 = np.zeros([resolution_x +  padsize, resolution_y + padsize, len(defocus_series)])
    images2 = np.pad(image,((int(padsize/2),int(padsize/2)),(int(padsize/2),int(padsize/2)),(0,0)),'edge')

    qx = makeFourierCoords1(resolution_x +  padsize,pixelsize)    
    qy = makeFourierCoords1(resolution_y +  padsize,pixelsize)
    [qxa,qya] = np.array(np.meshgrid(qx,qy, indexing='ij'))
    stackSize = [resolution_x + padsize, resolution_y + padsize, active]
    if ifaligned == False:
        dxy = np.zeros((stackSize[2], 2))
        dxy[:,0] = align_x
        dxy[:,1] = align_y

    for a0 in range(0, stackSize[2]):
        if ifaligned == False:
            psiShift =  np.exp(-2j*np.pi*pixelsize*(qxa*dxy[a0,0] + qya*dxy[a0,1]))
            a = np.fft.ifft2(np.fft.fft2(images2[:,:,a0]) * psiShift)
        else:
            a = images2[:,:,a0]
        a = np.float32(a)
        a1 = scipy.signal.wiener(a,[3,3])
        a1 = np.float32(a1)
        a1 = cv2.medianBlur(a1,5)
        a1 += cv2.Laplacian(a1,cv2.CV_32F,scale=0.5)
        a1 = cv2.normalize(a1,None,0,1,cv2.NORM_MINMAX)
        a = cv2.normalize(a,None,0,1,cv2.NORM_MINMAX)
        images1[:,:,a0] = a1
        images2[:,:,a0] = a 

    result_name = '/public/home/liuyang2022/TEM/result/im1.mat'
    io.savemat(result_name, {'im1':images1})
    result_name = '/public/home/liuyang2022/TEM/result/im2.mat'
    io.savemat(result_name, {'im2':images2})

    images2 = torch.Tensor(images2)
    images1 = torch.Tensor(images1.astype(np.float32))
    #加载模型，各神经网络
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
                                    
    model_sin = mlpNet(D=2,H=64,input_ch=inputch * level)
    model_sin = torch.nn.DataParallel(model_sin)
    model_sin = model_sin.to(device)
    optimizer_sin = torch.optim.Adam(model_sin.parameters(), 
                                    lr=5e-4, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-8, 
                                    weight_decay=0)

    model_cos = mlpNet(D=2,H=64,input_ch=inputch * level)
    model_cos = torch.nn.DataParallel(model_cos)
    model_cos = model_cos.to(device)
    optimizer_cos = torch.optim.Adam(model_cos.parameters(), 
                                    lr=5e-4, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-8, 
                                    weight_decay=0)
    
    #参数设置
    # criterion = MSE_and_GDL()
    criterion= torch.nn.HuberLoss()
    MSE=[]
    print('Begin')
    start = 0
    time0 = time.time()
    defocus_series = defocus_series[i_train]
    images1 = images1[:,:,i_train]
    images2 = images2[:,:,i_train]
    if imagetype == 'simulated':
        images1 = images2
    defocusrecord = np.zeros([len(defocus_series),100])
    defocusrecord[:,0] = defocus_series.cpu().numpy()



    #开始训练
    for i in trange(N_iters):
        

        selected_image = np.random.choice(len(i_train))
        k = int(selected_image)
        rangex1 = int(padsize/2)
        rangex2 = resolution_x + int(padsize/2)
        rangey1 = int(padsize/2)
        rangey2 = resolution_y + int(padsize/2)
        

        grid = torch.rand(1, resolution_x+padsize, resolution_y+padsize, 2)
        grid_input = (grid + offset) / ((resolution_x+padsize) / 2) - 1 

        input_mod = mlr_mod(grid_input)

        network_sin = model_sin(input_mod)
        network_res_real = network_sin.reshape([resolution_x + padsize, resolution_y + padsize])

        network_cos = model_cos(input_mod)
        network_res_imag = network_cos.reshape([resolution_x + padsize, resolution_y + padsize])

        img1 = network_res_real  + 1j * network_res_imag 

        defocus = defocus_series[k].clone().detach().requires_grad_(True)

        # img = fft.fft2(img1)
        # img = img * (ctfs_real[:,:,k] + 1j * ctfs_imag[:,:,k])
        # img = fft.ifft2(img)
        # img = torch.abs(img) ** 2


        img = MAL(defocus,resolution_x + padsize,resolution_y + padsize,pixelsize,img1,
                E0 = E0,Cs = Cs,alpha = alpha,M = 5,deltaC1 = deltaC1)
            
        #梯度清零
        if i <= N_iters *2/3 :

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

        loss = criterion(img[rangex1:rangex2,rangey1:rangey2], images1[rangex1:rangex2,rangey1:rangey2,k])
        MSE.append(float(loss))
        loss.backward()

        #更新神经网络中的权重      
        if i <= N_iters *2/3:
            optimizer_defocus.step()
        if i <= N_iters *4/5:
            optimizer_sin.step()
            optimizer_mlr_mod.step()
        optimizer_cos.step() 

        # print('defocus fitting:',defocus)
        # print('orignal defocus:',defocus_series[k])
        defocus_series[k] = defocus
        decay_steps = N_iters

        initial_lr = 5e-4
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

        # if (i + 1) % 2000 == 0:
        #     if i <= N_iters *1/2:
        #         for defocu in range(len(i_train)):
        #             defocus = defocus_series[defocu].clone()
        #             defoucs = defocusfitting(defocus,images1[rangex1:rangex2,rangey1:rangey2,k],
        #                 resolution_x + padsize,resolution_y + padsize,rangex1,rangex2,rangey1,rangey2,
        #                 pixelsize,img1,E0 = E0,Cs = Cs,alpha = alpha,M = 10,deltaC1 = deltaC1)
        #             defocus_series[k] = defocus

        #保存结果
        if (i + 1) % 2000 == 0:

            # defocusrecord[:,int((i + 1) / 2000)] = defocus_series.cpu().data.detach().numpy()
            print('defocus:',defocus_series)
        
        if (i + 1) % 20000 == 0:

            result_name = '/public/home/liuyang2022/TEM/result/MINIREALepoch' + diffstr + str(i+1) +'.mat'
            io.savemat(result_name, {'real': network_res_real.cpu().data.numpy().squeeze()})
            result_name = '/public/home/liuyang2022/TEM/result/MINIimagepoch' + diffstr + str(i+1) +'.mat'
            io.savemat(result_name, {'imag': network_res_imag.cpu().data.numpy().squeeze()})


            dt = time.time() - time0
            print(i,'/',N_iters,'iter  ', 'time: ', dt, 'loss: ', loss)
            time0 = time.time()
    # result_name = '/public/home/liuyang2022/TEM/result/defocusrecord' + diffstr + str(i+1) +'.mat'
    # io.savemat(result_name, {'defocus:',defocusrecord})
    result_name = '/public/home/liuyang2022/TEM/result/FS4epoch' + diffstr + str(N_iters) +'MSE.mat'
    io.savemat(result_name, {'loss':MSE})
    
    return
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()