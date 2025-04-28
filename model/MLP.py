# -*- coding: utf-8 -*-
import torch
import time
from scipy import io
import numpy as np
from utils.run_nerf_helpers import *
import cv2

class Network(torch.nn.Module):
    def __init__(self, D=8, H=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], no_rho=False):
        # 8 256 4
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.no_rho = no_rho

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, H)] + [torch.nn.Linear(H, H) if i not in self.skips else torch.nn.Linear(H + input_ch, H) for i in range(D-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + H, H//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if self.no_rho:
            self.output_linear = torch.nn.Linear(H, 1)
        else:
            self.feature_linear = torch.nn.Linear(H, H)
            self.alpha_linear = torch.nn.Linear(H, 1)
            self.rho_linear = torch.nn.Linear(H//2, 1)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # y_pred = self.linear(x)
        if self.no_rho:
            input_pts = x
            h = x
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.no_rho:
            outputs = self.output_linear(h)
            outputs = torch.abs(outputs)
        else:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.nn.functional.relu(h)

            rho = self.rho_linear(h)
            alpha = torch.abs(alpha)
            rho = torch.abs(rho)
            outputs = torch.cat([rho, alpha], -1)

        # if self.use_viewdirs:
        #     alpha = self.alpha_linear(h)
        #     feature = self.feature_linear(h)
        #     h = torch.cat([feature, input_views], -1)

        #     for i, l in enumerate(self.views_linears):
        #         h = self.views_linears[i](h)
        #         h = F.relu(h)

        #     rgb = self.rgb_linear(h)
        #     outputs = torch.cat([rgb, alpha], -1)
        # else:
        #     outputs = self.output_linear(h)

        return outputs

class PhaseNet(torch.nn.Module):
    def __init__(self, D=8, H=256, input_ch=3, skips=[4],tanh = False):
        # 8 256 4
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(PhaseNet, self).__init__()
        self.tanh = tanh
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, H)] + [torch.nn.Linear(H, H) if i not in self.skips else torch.nn.Linear(H + input_ch, H) for i in range(D-1)])

        # ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + H, H//2)])

        ### Implementation according to the paper
        self.linears = nn.ModuleList(
            [nn.Linear(input_ch + H, H)] + [nn.Linear(H, H)])

        # self.phase_linears = torch.nn.ModuleList(
        #     [nn.Linear(input_ch + H, H)] + [nn.Linear(H, H)])


        self.linear = torch.nn.Linear(H, 1)
        # self.phase_linear = torch.nn.Linear(H, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
                
        temp_h = torch.cat([x, h], -1)
        h = temp_h
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = torch.nn.functional.relu(h)

        modulus = self.linear(h)
        # if self.tanh:
        #     modulus = torch.nn.functional.tanh(modulus)
        # modulus = torch.abs(modulus)
        # h = temp_h
        # for i, l in enumerate(self.phase_linears):
        #     h = self.phase_linears[i](h)
        #     h = torch.nn.functional.relu(h)
        # phase = self.phase_linear(h)
        # outputs = torch.nn.functional.relu(outputs)
        # outputs = torch.cat([modulus, phase], axis = 1)

        outputs = modulus
        return outputs


class SCNet(torch.nn.Module):
    def __init__(self, D=8, H=256, input_ch=3, skips=[4]):
        # 8 256 4
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SCNet, self).__init__()
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, H)] + [torch.nn.Linear(H, H) if i not in self.skips else torch.nn.Linear(H + input_ch, H) for i in range(D-1)])

        self.mod_linears = nn.ModuleList(
            [nn.Linear(input_ch + H, H)] + [nn.Linear(H, H)])

        self.phase_linears = torch.nn.ModuleList(
            [nn.Linear(input_ch + H, H)] + [nn.Linear(H, H)])


        self.mod_linear = torch.nn.Linear(H, 1)
        self.phase_linear = torch.nn.Linear(H, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
                
        temp_h = torch.cat([x, h], -1)
        h = temp_h
        for i, l in enumerate(self.mod_linears):
            h = self.mod_linears[i](h)
            h = torch.nn.functional.relu(h)
        modulus = self.mod_linear(h)
   
        h = temp_h
        for i, l in enumerate(self.phase_linears):
            h = self.phase_linears[i](h)
            h = torch.nn.functional.relu(h)
        phase = self.phase_linear(h)
 
        outputs = torch.cat([modulus, phase], axis = 1)
        # outputs = torch.nn.functional.normalize(outputs,p=2,dim=2)
        return outputs

class mlpNet(nn.Module):
    def __init__(self, D=2, H=128, input_ch=6):
        super(mlpNet, self).__init__()
        self.input_ch = input_ch
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, H)] + [nn.Linear(H, H) for i in range(D)])

        self.linear = nn.Linear(H, 1)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        outputs = self.linear(h)

        return outputs

class Interpolation(torch.nn.Module):
    def __init__(self, in_channels, resolution):
        super(Interpolation, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.matrix = torch.nn.Parameter(torch.randn(in_channels, resolution, resolution))
    
    def forward(self, grid):
        output = torch.nn.functional.grid_sample(self.matrix[None], grid, mode='bilinear', padding_mode='border')
        return output

    
class MultilevelResolutionInterpolation(torch.nn.Module):
    def __init__(self, in_channels, level = 3):
        super(MultilevelResolutionInterpolation, self).__init__()
        self.in_channels = in_channels
        self.level = level
        self.matrices = torch.nn.ModuleList([Interpolation(in_channels, (i+1) * 64) for i in range(level)])
        
    def forward(self, grid):
        feature_list = []
        for i, l in enumerate(self.matrices):
            feature_list.append(self.matrices[i](grid))

        output = torch.cat(feature_list, dim=1)
        output = output[0].transpose(0, 2)
        output = output.reshape(-1, self.in_channels * self.level)
        return output

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # # Generate input points
    # input_x, input_y, input_z = np.meshgrid(
    #     np.linspace(0.0, 2.0, 20),
    #     np.linspace(0.0, 2.0, 20),
    #     np.linspace(0.0, 0.6, 6)
    # )
    # # print(input_x.shape)    test_input = np.concatenate((input_x.reshape([-1, 1]), input_y.reshape([-1, 1]), input_z.reshape([-1, 1])), axis = 1)
    # test_input = torch.Tensor(test_input).to(device)
    # print(f'Input: {test_input.shape}')
    # test_input = encoding_sph_tensor(test_input, 10, 10, True)
    # print(f'Encoded input: {test_input.shape}')

    # Load input points
    # test_input = io.loadmat('./data/node_coordinates.mat')['node']
    sample_pts_x, sample_pts_y = torch.meshgrid(
        torch.linspace(-256 / 2, 256 / 2, 256),
        torch.linspace(-256 / 2, 256/ 2, 256),
    )
    sample_pts_x = sample_pts_x.reshape([-1, 1])
    sample_pts_y = sample_pts_y.reshape([-1, 1])
    test_input = torch.cat([sample_pts_x, sample_pts_y], axis = 1) / 128
    test_input = encoding(test_input, 10)
    print(f'Test input: {test_input.shape}')
 
    # Load target 
    # target = io.loadmat('./data/mua.mat')['mua']
    # target = hdf5storage.loadmat('./data/mua.mat')['mua']
    # target = torch.Tensor(target).to(device)
    # print(f'Target: {target.shape}')

    # img = cv2.imread('./data/focal_simulation/Phase of exit wave.tif',-1)
    path = "./data/focal_simulation/" 
    phase_gt_name = 'Phase of exit wave.tif'
    phase_gt = cv2.imread(path+ phase_gt_name,-1)
    phase_gt = np.array(phase_gt)
    modulus_gt_name = 'Modulus of exit wave.tif'
    modulus_gt = cv2.imread(path+ modulus_gt_name,-1)
    modulus_gt = np.array(modulus_gt)
    img_gt = modulus_gt * np.e**(1j*phase_gt)
    img = np.fft.fft2(img_gt)
    img = np.fft.fftshift(img)
    modulus = np.real(img) / 150000
    imag = np.imag(img) / 150000
    modulus = modulus.reshape([-1, 1])
    imag = imag.reshape([-1, 1])
    print(modulus.max(), modulus.min(), imag.max(), imag.min())
    img = np.concatenate([modulus, imag], axis = 1)
    io.savemat('./represent/target.mat', {'target':img}) 

    target = torch.Tensor(img).to(device)
    # target = target.reshape([-1, 1])
    print(f'Target: {target.shape}')

    # print('Start regression.')
    model = PhaseNet(D=8, H=256, input_ch=4 * 10, skips=[4])
    model.to(device)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()

    print('Start regression.')

    num_iters = 2000
    # # temp_input = test_input
    # # test_input = encoding_sph_tensor(test_input, 10, 10, True)

    time0 = time.time()
    for i in range(num_iters):

        # index_rand = torch.randint(0, target.shape[0], (index_num,))
        net_res = model(test_input)
        # print(net_res.dtype)
        # loss = criterion(net_res[index_fine], target[index_fine]) + criterion(net_res[index_rand], target[index_rand])
        # print(net_res.shape, target.shape)
        loss = criterion(target, net_res)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # if (i+1) % 50 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.998

        if (i+1) % 100 == 0:
            # model_name = './model/epoch' + str(i+1) + '.pt'
            # torch.save(model, model_name) 
            io.savemat('./represent/epoch' + str(i+1) + '.mat', {'phase':net_res.cpu().data.numpy().squeeze()}) 

        if (i) % 100 == 0:
            dt = time.time()-time0
            print(i,'/',num_iters,'iter  ', 'time: ', dt, 'loss: ', loss)
            time0 = time.time()


