import torch
from torch.optim import Adam
from models import MultilevelResolutionInterpolation, mlpNet
from loss import SSIM
import time
import numpy as np
from utils import makeFourierCoords, focalSeries, MAL, MAL_Astigmatism

def initialize_models(inputch, level, device):
    """
    初始化多分辨率插值模型和正弦余弦模型

    参数:
        inputch (int): 输入通道数
        level (int): 分辨率级别
        device (torch.device): 设备类型 (cuda 或 cpu)
    
    返回:
        mlr_mod (torch.nn.DataParallel): 多分辨率插值模型
        model_sin (torch.nn.DataParallel): 正弦模型
        model_cos (torch.nn.DataParallel): 余弦模型
    """
    mlr_mod = MultilevelResolutionInterpolation(inputch, level)
    mlr_mod = torch.nn.DataParallel(mlr_mod).to(device)
    
    model_sin = mlpNet(D=3, H=64, input_ch=inputch * level)
    model_sin = torch.nn.DataParallel(model_sin).to(device)
    
    model_cos = mlpNet(D=3, H=64, input_ch=inputch * level)
    model_cos = torch.nn.DataParallel(model_cos).to(device)
    
    return mlr_mod, model_sin, model_cos

def initialize_optimizers(mlr_mod, model_sin, model_cos, lr=5e-4):
    """
    初始化优化器
    
    参数:
        mlr_mod (torch.nn.Module): 多分辨率插值模型
        model_sin (torch.nn.Module): 正弦模型
        model_cos (torch.nn.Module): 余弦模型
        lr (float): 学习率
    
    返回:
        optimizer_mlr_mod (torch.optim.Adam): 多分辨率插值模型优化器
        optimizer_sin (torch.optim.Adam): 正弦模型优化器
        optimizer_cos (torch.optim.Adam): 余弦模型优化器
    """
    optimizer_mlr_mod = Adam(mlr_mod.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer_sin = Adam(model_sin.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer_cos = Adam(model_cos.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
    return optimizer_mlr_mod, optimizer_sin, optimizer_cos

def train_one_epoch(i, N_iters, mlr_mod, model_sin, model_cos, optimizer_mlr_mod, optimizer_sin, optimizer_cos, images, defocus_series, resolution_x, resolution_y, padsize, offset, criterion, ifAstigmatism, E0, Cs, alpha, M, deltaC1):
    """
    训练每一轮的代码
    
    参数:
        i (int): 当前迭代轮次
        N_iters (int): 总迭代次数
        mlr_mod (torch.nn.Module): 多分辨率插值模型
        model_sin (torch.nn.Module): 正弦模型
        model_cos (torch.nn.Module): 余弦模型
        optimizer_mlr_mod (torch.optim.Adam): 多分辨率插值模型优化器
        optimizer_sin (torch.optim.Adam): 正弦模型优化器
        optimizer_cos (torch.optim.Adam): 余弦模型优化器
        images (torch.Tensor): 图像数据
        defocus_series (torch.Tensor): 焦距序列
        resolution_x (int): 图像分辨率的 x 方向
        resolution_y (int): 图像分辨率的 y 方向
        padsize (int): 填充大小
        offset (torch.Tensor): 偏移量
        criterion (callable): 损失函数（SSIM）
        ifAstigmatism (bool): 是否考虑像差
        E0 (float): 入射电子能量
        Cs (float): 像差常数
        alpha (float): 镜头参数
        M (int): 放大倍数
        deltaC1 (float): 透镜球差
    """
    selected_image = np.random.choice(len(i_train))
    k = int(selected_image)
    
    grid = torch.rand(1, resolution_x + padsize, resolution_y + padsize, 2)
    grid_input = (offset + grid) / ((resolution_x + padsize) / 2) - 1
    input_mod = mlr_mod(grid_input)
    
    network_sin = model_sin(input_mod)
    network_res_real = network_sin.reshape([resolution_x + padsize, resolution_y + padsize])
    
    network_cos = model_cos(input_mod)
    network_res_imag = network_cos.reshape([resolution_x + padsize, resolution_y + padsize])
    
    img1 = network_res_real + 1j * network_res_imag
    defocus = defocus_series[k].clone().detach().requires_grad_(True)
    
    if ifAstigmatism:
        img = MAL_Astigmatism(defocus, resolution_x + padsize, resolution_y + padsize, pixelsize, img1, E0=E0, Cs=Cs, alpha=alpha, M=M, deltaC1=deltaC1, theta=0, z=150)
    else:
        img = MAL(defocus, resolution_x + padsize, resolution_y + padsize, pixelsize, img1, E0=E0, Cs=Cs, alpha=alpha, M=M, deltaC1=deltaC1)
    
    img = img.unsqueeze(0).unsqueeze(0)
    image1 = images[44:1044, 44:1044, k]
    image1 = image1.unsqueeze(0).unsqueeze(0)
    
    loss = 1 - criterion(img[:, :, 44:1044, 44:1044], image1)
    loss.backward()
    
    if i <= N_iters * 2 / 3 and ifdefoucsfitting != False:
        optimizer_defocus = Adam([defocus], lr=5e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        optimizer_defocus.zero_grad()
        optimizer_defocus.step()
    
    if i <= N_iters * 4 / 5:
        optimizer_sin.zero_grad()
        optimizer_mlr_mod.zero_grad()
    optimizer_cos.zero_grad()
    
    if i <= N_iters * 4 / 5:
        optimizer_sin.step()
        optimizer_mlr_mod.step()
    optimizer_cos.step()
    
    return loss.item()

def train(model_params, data_params, training_params):
    """
    训练主函数
    
    参数:
        model_params (dict): 模型参数
        data_params (dict): 数据参数
        training_params (dict): 训练超参数
    """
    device = model_params['device']
    inputch = model_params['inputch']
    level = model_params['level']
    
    images = data_params['images']
    defocus_series = data_params['defocus_series']
    
    N_iters = training_params['N_iters']
    resolution_x = training_params['resolution_x']
    resolution_y = training_params['resolution_y']
    padsize = training_params['padsize']
    offset = training_params['offset']
    criterion = SSIM()
    
    mlr_mod, model_sin, model_cos = initialize_models(inputch, level, device)
    optimizer_mlr_mod, optimizer_sin, optimizer_cos = initialize_optimizers(mlr_mod, model_sin, model_cos)
    
    for i in range(N_iters):
        loss = train_one_epoch(i, N_iters, mlr_mod, model_sin, model_cos, optimizer_mlr_mod, optimizer_sin, optimizer_cos, images, defocus_series, resolution_x, resolution_y, padsize, offset, criterion, model_params['ifAstigmatism'], model_params['E0'], model_params['Cs'], model_params['alpha'], model_params['M'], model_params['deltaC1'])
        
        # 每 1000 次迭代打印一次损失
        if i % 1000 == 0:
            print(f"Iteration {i}/{N_iters}, Loss: {loss}")

    print("Training finished!")

if __name__ == '__main__':
    # 这里需要传入模型参数、数据参数和训练参数
    model_params = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'inputch': 4,
        'level': 32,
        'ifAstigmatism': True,
        'E0': 200,  # 示例
        'Cs': 1.0,  # 示例
        'alpha': 0.1,  # 示例
        'M': 5,  # 示例
        'deltaC1': 0.01,  # 示例
    }
    
    data_params = {
        'images': torch.randn(1, 1000, 1000),  # 示例
        'defocus_series': torch.randn(1, 1000),  # 示例
    }
    
    training_params = {
        'N_iters': 10000,
        'resolution_x': 1000,
        'resolution_y': 1000,
        'padsize': 16,
        'offset': torch.randn(1, 1000, 1000, 2),  # 示例
    }
    
    train(model_params, data_params, training_params)