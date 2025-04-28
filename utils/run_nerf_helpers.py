import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# TODO: remove this dependency
# from torchsearchsorted import searchsorted


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # # Get pdf
    # weights = weights + 1e-5 # prevent nans
    # pdf = weights / torch.sum(weights, -1, keepdim=True)
    # cdf = torch.cumsum(pdf, -1)
    # cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # # Take uniform samples
    # if det:
    #     u = torch.linspace(0., 1., steps=N_samples)
    #     u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    # else:
    #     u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # # Pytest, overwrite u with numpy's fixed random numbers
    # if pytest:
    #     np.random.seed(0)
    #     new_shape = list(cdf.shape[:-1]) + [N_samples]
    #     if det:
    #         u = np.linspace(0., 1., N_samples)
    #         u = np.broadcast_to(u, new_shape)
    #     else:
    #         u = np.random.rand(*new_shape)
    #     u = torch.Tensor(u)

    # # Invert CDF
    # u = u.contiguous()
    # inds = searchsorted(cdf, u, side='right')
    # below = torch.max(torch.zeros_like(inds-1), inds-1)
    # above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    # inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # denom = (cdf_g[...,1]-cdf_g[...,0])
    # denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    # t = (u-cdf_g[...,0])/denom
    # samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

# Spherical Sampling
def spherical_sample(t, x0, y0, z0, accurate_sampling, volume_position, volume_size, sampling_points, no_view):
    # [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]
    # v_light = 3 * 1e8
    # r = v_light * 4 * t * 1e-12

    v_light = 1
    r = v_light * 0.003 * t

    if accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)
        theta_min = np.min(sphere_box_point[:,1])
        theta_max = np.max(sphere_box_point[:,1])
        phi_min = np.min(sphere_box_point[:,2])
        phi_max = np.max(sphere_box_point[:,2])
        theta = np.linspace(theta_min, theta_max , sampling_points)
        phi = np.linspace(phi_min, phi_max, sampling_points)
        # 预设采样范围
        # box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        # box_point[:,0] = box_point[:,0] - x0
        # box_point[:,1] = box_point[:,1] - y0
        # box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        # sphere_box_point = cartesian2spherical(box_point)

        # [xv, yv, zv] = [volume_position[0], volume_position[1], volume_position[2]]
        # # xv, yv, zv 是物体 volume 的中心坐标
        # dx = volume_size[0]
        # dy = volume_size[1]
        # dz = volume_size[2]

        # phi_min = np.arccos((xv - dx - x0) / np.sqrt((xv - dx - x0)**2 + (yv - dy - y0)**2))
        # phi_max = np.arccos((xv + dx - x0) / np.sqrt((xv + dx - x0)**2 + (yv - dy - y0)**2))
        # # phi_min = np.arccos((zv - dz - z0) / np.sqrt((zv - dz - z0)**2 + (yv - dy)**2))
        # # phi_max = np.arccos((zv + dz - z0) / np.sqrt((zv + dz - z0)**2 + (yv - dy)**2))     
        # theta_min = np.arccos((zv - dz - z0) / np.sqrt((zv - dz - z0)**2 + (yv - dy - y0)**2))
        # theta_max = np.arccos((zv + dz - z0) / np.sqrt((zv + dz - z0)**2 + (yv - dy - y0)**2))  

        # theta = np.linspace(theta_min, theta_max , 18)
        # phi = np.linspace(phi_min, phi_max, 18)
    else:
        # k = 16
        # theta = np.linspace(0 + np.pi / 32, np.pi - np.pi / 32, 16)
        # phi = np.linspace(-np.pi +np.pi / 32 , 0 - np.pi / 32, 16)
        theta = np.linspace(0, np.pi, sampling_points)
        # phi = np.linspace(-np.pi , 0 , 19)
        phi = np.linspace(0 , np.pi, sampling_points)
    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [-pi,0]

    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    thetas = np.outer(theta, np.ones_like(phi))
    phis = np.outer(np.ones_like(phi), phi)
    # print(x.shape)
    x = x.flatten().reshape([-1,1])
    y = y.flatten().reshape([-1,1])
    z = z.flatten().reshape([-1,1])
    thetas = thetas.flatten().reshape([-1,1])
    phis = phis.flatten().reshape([-1,1])

    if no_view:
        samples = np.concatenate((x*r + x0, y*r + y0, z*r + z0), axis=1)
    else:
        samples = np.concatenate((x*r + x0, y*r + y0, z*r + z0, thetas, phis), axis=1)

    return samples # 注意：如果sampling正确的话，x 和 z 应当关于 0 对称， y 应当只有负值

# Spherical Sampling
def spherical_sample_histgram(camera_grid_positions, num_sampling_points, test_accurate_sampling, volume_position, volume_size, c, deltaT, no_rho, start, end):
    # t = np.linspace(I,L,L - I + 1) * deltaT # 
    # 输入: camera_grid_points, I
    # 输出：(L - I + 1) x 256 x 3
    [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]
    # v_light = c #3 * 1e8
    # r = v_light * t 

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)
        # r_min = np.min(sphere_box_point[:,0])
        # r_min = 0.1 # zaragoza 256
        # r_min = 0.35 # generated data
        # r_min = min(r_min, abs(volume_position[1] + volume_size), 0.15)
        
        # r_max = np.max(sphere_box_point[:,0]) + 0
        # r_max = min(r_max, (L - 1) * v_light * deltaT)

        r_min = 100 * c * deltaT
        r_max = 300 * c * deltaT
        theta_min = np.min(sphere_box_point[:,1]) - 0
        theta_max = np.max(sphere_box_point[:,1]) + 0
        phi_min = np.min(sphere_box_point[:,2]) - 0
        phi_max = np.max(sphere_box_point[:,2]) + 0
        theta = np.linspace(theta_min, theta_max , num_sampling_points)
        phi = np.linspace(phi_min, phi_max, num_sampling_points)
        dtheta = (theta_max - theta_min) / num_sampling_points
        dphi = (phi_max - phi_min) / num_sampling_points
        # 预设采样范围
    else:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)

        r_min = 0.1
        r_max = np.max(sphere_box_point[:,0])
        # theta = np.linspace(0, np.pi , num_sampling_points)
        # phi = np.linspace(-np.pi, 0, num_sampling_points)
        # theta = np.linspace(0.1, np.pi - 0.1 , num_sampling_points)
        # phi = np.linspace(-np.pi + 0.1, 0 - 0.1, num_sampling_points)
        theta = np.linspace(0, np.pi , num_sampling_points)
        phi = np.linspace(-np.pi, 0, num_sampling_points)

        dtheta = (np.pi) / num_sampling_points
        dphi = (np.pi) / num_sampling_points
    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [-pi,0]

    # r_min = 130 * c * deltaT # zaragoza64 bunny
    # r_max = 300 * c * deltaT

    # r_min = 100 * c * deltaT # zaragoza256 bunny # fk 和 
    # r_max = 300 * c * deltaT

    # r_min = 1 * c * deltaT # serapis
    # r_max = 500 * c * deltaT

    # r_min = 300 * c * deltaT # zaragoza256 T 
    # r_max = 500 * c * deltaT

    # r_min = 250 * c * deltaT # zaragoza256_2 
    # r_max = 450 * c * deltaT

    r_min = start * c * deltaT # zaragoza256_2 
    r_max = end * c * deltaT

    num_r = end - start
    r = np.linspace(r_min, r_max , num_r)

    I1 = r_min / (c * deltaT)
    I2 = r_max / (c * deltaT)

    I1 = math.floor(I1)
    I2 = math.ceil(I2)
    I0 = r.shape[0]

    '''x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    # print(x.shape)
    x = x.flatten().reshape([-1,1])
    y = y.flatten().reshape([-1,1])
    z = z.flatten().reshape([-1,1])
    samples = np.concatenate((x*r + x0, y*r + y0, z*r + z0), axis=1)'''

    grid = np.stack(np.meshgrid(r, theta, phi),axis = -1)
    grid = np.transpose(grid,(1,0,2,3))

    spherical = grid.reshape([-1,3])
    cartesian = spherical2cartesian(spherical)
    cartesian = cartesian + np.array([x0,y0,z0])
    if not no_rho:
        cartesian = np.concatenate((cartesian, spherical[:,1:3]), axis = 1)
    # cartesian_grid = cartesian.reshape(grid.shape)
    # cartesian_grid = cartesian_grid.reshape([L - I + 1, num_sampling_points ** 2, 3])
    #  
    # print(I1,I1+I0)
    return cartesian, dtheta, dphi  # 注意：如果sampling正确的话，x 和 z 应当关于 x0,z0 对称， y 应当只有负值

# Spherical Sampling
def spherical_sample_histgram_tensor(camera_grid_positions, num_sampling_points, test_accurate_sampling, volume_position, volume_size, c, deltaT, no_rho, start, end):
    # t = np.linspace(I,L,L - I + 1) * deltaT # 
    # 输入: camera_grid_points, I
    # 输出：(L - I + 1) x 256 x 3
    [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]
    # v_light = c #3 * 1e8
    # r = v_light * t 

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0.cpu().numpy()
        box_point[:,1] = box_point[:,1] - y0.cpu().numpy()
        box_point[:,2] = box_point[:,2] - z0.cpu().numpy() # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)
        # r_min = np.min(sphere_box_point[:,0])
        # r_min = 0.1 # zaragoza 256
        # r_min = 0.35 # generated data
        # r_min = min(r_min, abs(volume_position[1] + volume_size), 0.15)
        
        # r_max = np.max(sphere_box_point[:,0]) + 0
        # r_max = min(r_max, (L - 1) * v_light * deltaT)

        r_min = 100 * c * deltaT
        r_max = 300 * c * deltaT
        theta_min = np.min(sphere_box_point[:,1]) - 0
        theta_max = np.max(sphere_box_point[:,1]) + 0
        phi_min = np.min(sphere_box_point[:,2]) - 0
        phi_max = np.max(sphere_box_point[:,2]) + 0
        theta = torch.linspace(theta_min, theta_max , num_sampling_points)
        phi = torch.linspace(phi_min, phi_max, num_sampling_points)
        dtheta = (theta_max - theta_min) / num_sampling_points
        dphi = (phi_max - phi_min) / num_sampling_points
        # 预设采样范围
    else:
        # box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        # box_point[:,0] = box_point[:,0] - x0
        # box_point[:,1] = box_point[:,1] - y0
        # box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        # sphere_box_point = cartesian2spherical(box_point)

        # r_min = 0.1
        # r_max = np.max(sphere_box_point[:,0])

        # theta = np.linspace(0, np.pi , num_sampling_points)
        # phi = np.linspace(-np.pi, 0, num_sampling_points)
        # theta = np.linspace(0.1, np.pi - 0.1 , num_sampling_points)
        # phi = np.linspace(-np.pi + 0.1, 0 - 0.1, num_sampling_points)
        theta = torch.linspace(0, np.pi , num_sampling_points).cuda()
        phi = torch.linspace(-np.pi, 0, num_sampling_points).cuda()
        
        theta_min = 0
        theta_max = np.pi
        phi_min = -np.pi
        phi_max = 0

        dtheta = (np.pi) / num_sampling_points
        dphi = (np.pi) / num_sampling_points
    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [-pi,0]

    # r_min = 130 * c * deltaT # zaragoza64 bunny
    # r_max = 300 * c * deltaT

    # r_min = 100 * c * deltaT # zaragoza256 bunny # fk 和 
    # r_max = 300 * c * deltaT

    # r_min = 1 * c * deltaT # serapis
    # r_max = 500 * c * deltaT

    # r_min = 300 * c * deltaT # zaragoza256 T 
    # r_max = 500 * c * deltaT

    # r_min = 250 * c * deltaT # zaragoza256_2 
    # r_max = 450 * c * deltaT

    r_min = start * c * deltaT # zaragoza256_2 
    r_max = end * c * deltaT

    num_r = end - start
    r = torch.linspace(r_min, r_max , num_r)

    I1 = r_min / (c * deltaT)
    I2 = r_max / (c * deltaT)

    I1 = math.floor(I1)
    I2 = math.ceil(I2)
    # I0 = r.shape[0]

    '''x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    # print(x.shape)
    x = x.flatten().reshape([-1,1])
    y = y.flatten().reshape([-1,1])
    z = z.flatten().reshape([-1,1])
    samples = np.concatenate((x*r + x0, y*r + y0, z*r + z0), axis=1)'''

    # grid = np.stack(np.meshgrid(r, theta, phi),axis = -1)
    # grid = np.transpose(grid,(1,0,2,3))

    # spherical = grid.reshape([-1,3])
    # cartesian = spherical2cartesian(spherical)
    # cartesian = cartesian + np.array([x0,y0,z0])
    # if not no_rho:
    #     cartesian = np.concatenate((cartesian, spherical[:,1:3]), axis = 1)
    grid = torch.stack(torch.meshgrid(r, theta, phi), axis = -1)
    # grid = torch.transpose(grid, 1, 0)

    spherical = grid.reshape([-1,3])
    cartesian = spherical2cartesian(spherical)
    cartesian = cartesian + torch.Tensor([x0,y0,z0])
    if not no_rho:
        cartesian = torch.cat((cartesian, spherical[:,1:3]), axis = 1)
    # cartesian_grid = cartesian.reshape(grid.shape)
    # cartesian_grid = cartesian_grid.reshape([L - I + 1, num_sampling_points ** 2, 3])
    #  
    # print(I1,I1+I0)
    # return cartesian, dtheta, dphi, theta_max, theta_min, phi_max, phi_min  # 注意：如果sampling正确的话，x 和 z 应当关于 x0,z0 对称， y 应当只有负值
    return cartesian, dtheta, dphi
    
# Spherical Sampling
def spherical_sample_real_test(camera_grid_positions, num_sampling_points, test_accurate_sampling, volume_position, volume_size, c, deltaT, no_rho, start, end):
    # t = np.linspace(I,L,L - I + 1) * deltaT # 
    # 输入: camera_grid_points, I
    # 输出：(L - I + 1) x 256 x 3
    [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]
    # v_light = c #3 * 1e8
    # r = v_light * t 

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)
        # r_min = np.min(sphere_box_point[:,0])
        # r_min = 0.1 # zaragoza 256
        # r_min = 0.35 # generated data
        # r_min = min(r_min, abs(volume_position[1] + volume_size), 0.15)
        
        # r_max = np.max(sphere_box_point[:,0]) + 0
        # r_max = min(r_max, (L - 1) * v_light * deltaT)

        r_min = 100 * c * deltaT
        r_max = 300 * c * deltaT
        theta_min = np.min(sphere_box_point[:,1]) - 0
        theta_max = np.max(sphere_box_point[:,1]) + 0
        phi_min = np.min(sphere_box_point[:,2]) - 0
        phi_max = np.max(sphere_box_point[:,2]) + 0
        theta = np.linspace(theta_min, theta_max , num_sampling_points)
        phi = np.linspace(phi_min, phi_max, num_sampling_points)
        dtheta = (theta_max - theta_min) / num_sampling_points
        dphi = (phi_max - phi_min) / num_sampling_points
        # 预设采样范围
    else:
        box_point = volume_box_point(volume_position, volume_size) # 返回物体box的八个顶点的直角坐标
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # 这三行考虑了原点在相机位置上的问题
        sphere_box_point = cartesian2spherical(box_point)

        r_min = 0.1
        r_max = np.max(sphere_box_point[:,0])
        # theta = np.linspace(0, np.pi , num_sampling_points)
        # phi = np.linspace(-np.pi, 0, num_sampling_points)
        # theta = np.linspace(0.1, np.pi - 0.1 , num_sampling_points)
        # phi = np.linspace(-np.pi + 0.1, 0 - 0.1, num_sampling_points)
        theta = np.linspace(0, np.pi , num_sampling_points)
        phi = np.linspace(-np.pi, 0, num_sampling_points)

        dtheta = (np.pi) / num_sampling_points
        dphi = (np.pi) / num_sampling_points
    # 直角坐标图像参考 Zaragoza 数据集中的坐标系
    # 球坐标图像参考 Wikipedia 球坐标系词条 ISO 约定
    # theta 是 俯仰角，与 Z 轴正向的夹角， 范围从 [0,pi]
    # phi 是 在 XOY 平面中与 X 轴正向的夹角， 范围从 [-pi,pi],本场景中只用到 [-pi,0]

    # r_min = 130 * c * deltaT # zaragoza64 bunny
    # r_max = 300 * c * deltaT

    # r_min = 100 * c * deltaT # zaragoza256 bunny # fk 和 
    # r_max = 300 * c * deltaT

    # r_min = 1 * c * deltaT # serapis
    # r_max = 500 * c * deltaT

    # r_min = 300 * c * deltaT # zaragoza256 T 
    # r_max = 500 * c * deltaT

    # r_min = 250 * c * deltaT # zaragoza256_2 
    # r_max = 450 * c * deltaT

    r_min = start * c * deltaT # zaragoza256_2 
    r_max = end * c * deltaT

    num_r = end - start
    r = np.linspace(r_min, r_max , num_r)

    I1 = r_min / (c * deltaT)
    I2 = r_max / (c * deltaT)

    I1 = math.floor(I1)
    I2 = math.ceil(I2)
    I0 = r.shape[0]

    '''x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    # print(x.shape)
    x = x.flatten().reshape([-1,1])
    y = y.flatten().reshape([-1,1])
    z = z.flatten().reshape([-1,1])
    samples = np.concatenate((x*r + x0, y*r + y0, z*r + z0), axis=1)'''

    grid = np.stack(np.meshgrid(r, theta, phi),axis = -1)
    grid = np.transpose(grid,(1,0,2,3))

    spherical = grid.reshape([-1,3])
    cartesian = spherical2cartesian(spherical)
    cartesian = cartesian + np.array([x0,y0,z0])
    if not no_rho:
        cartesian = np.concatenate((cartesian, spherical[:,1:3]), axis = 1)
    # cartesian_grid = cartesian.reshape(grid.shape)
    # cartesian_grid = cartesian_grid.reshape([L - I + 1, num_sampling_points ** 2, 3])
    #  
    # print(I1,I1+I0)
    return cartesian, dtheta, dphi  # 注意：如果sampling正确的话，x 和 z 应当关于 x0,z0 对称， y 应当只有负值
    
def nomalize(hist, xyzmin, xyzmax):

    return kkk

def encoding(pt, L):
    # coded_pt = torch.zeros(6 * L)
    # logseq = torch.logspace(start=0, end=L-1, steps=L, base=2)
    # xsin = torch.sin(logseq.mul(math.pi).mul(pt[0]))
    # ysin = torch.sin(logseq.mul(math.pi).mul(pt[1]))
    # zsin = torch.sin(logseq.mul(math.pi).mul(pt[2]))
    # xcos = torch.cos(logseq.mul(math.pi).mul(pt[0]))
    # ycos = torch.cos(logseq.mul(math.pi).mul(pt[1]))
    # zcos = torch.cos(logseq.mul(math.pi).mul(pt[2]))
    # coded_pt = torch.reshape(torch.cat((xsin,xcos,ysin,ycos,zsin,zcos)), (1, 6 * L))

    logseq = np.logspace(start=0, stop=L-1, num=L, base=2)
    xsin = np.sin(logseq*math.pi*pt[0])
    ysin = np.sin(logseq*math.pi*pt[1])
    zsin = np.sin(logseq*math.pi*pt[2])
    xcos = np.cos(logseq*math.pi*pt[0])
    ycos = np.cos(logseq*math.pi*pt[1])
    zcos = np.cos(logseq*math.pi*pt[2])
    coded_pt = np.reshape(np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos)), (1, 6 * L))
    # i = 1
    return coded_pt

def encoding_sph(hist, L, L_view, no_view):
    # coded_hist = torch.cat([encoding(hist[k], L) for k in range(hist.shape[0])], 0)
    logseq = np.logspace(start=0, stop=L-1, num=L, base=2)
    logseq_view = np.logspace(start=0, stop=L_view-1, num=L_view, base=2)

    xsin = np.sin((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ysin = np.sin((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zsin = np.sin((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    xcos = np.cos((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ycos = np.cos((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zcos = np.cos((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    if no_view:
        coded_hist = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos), axis=1)
    else:
        thetasin = np.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phisin = np.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        thetacos = np.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phicos = np.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        coded_hist = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos), axis=1)
        # coded_hist = np.concatenate([encoding(hist[k], L) for k in range(hist.shape[0])], axis=0)

    return coded_hist

def encoding_sph_tensor(hist, L, L_view, no_view):
    # coded_hist = torch.cat([encoding(hist[k], L) for k in range(hist.shape[0])], 0)
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2).float().cuda()
    logseq_view = torch.logspace(start=0, end=L_view-1, steps=L_view, base=2).float().cuda()

    xsin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ysin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zsin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    xcos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ycos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zcos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    if no_view:
        coded_hist = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos), axis=1)
    else:
        thetasin = torch.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phisin = torch.sin((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        thetacos = torch.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,3].reshape([-1, 1]))
        phicos = torch.cos((logseq_view*math.pi).reshape([1,-1])*hist[:,4].reshape([-1, 1]))
        coded_hist = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos), axis=1)
        # coded_hist = np.concatenate([encoding(hist[k], L) for k in range(hist.shape[0])], axis=0)

    return coded_hist

def encoding(hist, L):
    # coded_hist = torch.cat([encoding(hist[k], L) for k in range(hist.shape[0])], 0)
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2).float().cuda()

    xsin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ysin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    xcos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ycos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))

    coded_hist = torch.cat((xsin,xcos,ysin,ycos), axis=1)

        # coded_hist = np.concatenate([encoding(hist[k], L) for k in range(hist.shape[0])], axis=0)

    return coded_hist

def positional_encoding(hist, L):
    # coded_hist = torch.cat([encoding(hist[k], L) for k in range(hist.shape[0])], 0)
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2).float().cuda()

    xsin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ysin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zsin = torch.sin((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))
    xcos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,0].reshape([-1, 1]))
    ycos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,1].reshape([-1, 1]))
    zcos = torch.cos((logseq*math.pi).reshape([1,-1])*hist[:,2].reshape([-1, 1]))

    coded_hist = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos), axis=1)
    return coded_hist

def nlos_render(samples, camera_gridposition, model, use_encoding, encoding_dim,volume_position, volume_size):
    sphere_res = 0 # sphere_res 是nerf渲染出的单个bin的结果， 是一个torch.Tensor标量
    distance_square = camera_gridposition[0] ** 2 + camera_gridposition[1] ** 2 + camera_gridposition[2] ** 2
    for l in range(len(samples[0])):
        x = samples[l, 0] + volume_size[0] / (2 * volume_size[0])
        y = samples[l, 1] - volume_position[1] / (2 * volume_size[0])
        z = samples[l, 2] + volume_size[0] / (2 * volume_size[0])
        pt = torch.tensor([x, y, z], dtype=torch.float32).view(-1)
        # print(pt)
        if use_encoding:
            coded_pt = encoding(pt, L=encoding_dim) # encoding 函数将长度为 3 的 tensor 返回为长度为 6L 的tensor
            network_res = model(coded_pt)
        else:
            network_res = model(pt)
        sphere_res = sphere_res + network_res[0] / distance_square

    return sphere_res

def show_samples(samples,volume_position,volume_size):
    box = volume_box_point(volume_position,volume_size)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(0,0,0,c='g',linewidths=0.03)
    ax.scatter(box[:,0],box[:,1],box[:,2], c = 'b', linewidths=0.03)
    ax.scatter(volume_position[0],volume_position[1],volume_position[2],c = 'b', linewidths=0.03)
    ax.scatter(samples[:,0],samples[:,1],samples[:,2],c='r',linewidths=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig('./scatter_samples')
    plt.close()

    plt.scatter(0,0,c='g',linewidths=0.03)
    plt.scatter(box[:,0],box[:,1], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[1],c = 'b', linewidths=0.03)
    plt.scatter(samples[:,0],samples[:,1],c = 'r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.savefig('./scatter_samples_XOY')
    plt.close()

    plt.scatter(0,0,c='g',linewidths=0.03)
    plt.scatter(box[:,0],box[:,2], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[2],c = 'b', linewidths=0.03)
    plt.scatter(samples[:,0],samples[:,2], c = 'r')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.show()
    plt.savefig('./scatter_samples_XOZ')
    plt.close()

    return 0

def volume_box_point(volume_position, volume_size):
    [xv, yv, zv] = [volume_position[0], volume_position[1], volume_position[2]]
    # xv, yv, zv 是物体 volume 的中心坐标
    dx = volume_size[0]
    dy = volume_size[0]
    dz = volume_size[0]
    x = np.concatenate((xv - dx, xv - dx, xv - dx, xv - dx, xv + dx, xv + dx, xv + dx, xv + dx), axis=0).reshape([-1, 1])
    y = np.concatenate((yv - dy, yv - dy, yv + dy, yv + dy, yv - dy, yv - dy, yv + dy, yv + dy), axis=0).reshape([-1, 1])
    z = np.concatenate((zv - dz, zv + dz, zv - dz, zv + dz, zv - dz, zv + dz, zv - dz, zv + dz), axis=0).reshape([-1, 1])
    box = np.concatenate((x, y, z),axis = 1)
    return box

def cartesian2spherical(pt):
    # 函数将直角坐标系下的点转换为球坐标系下的点
    # 输入格式： pt 是一个 N x 3 的 ndarray

    spherical_pt = np.zeros(pt.shape)
    spherical_pt[:,0] = np.sqrt(np.sum(pt ** 2,axis=1))
    spherical_pt[:,1] = np.arccos(pt[:,2] / spherical_pt[:,0])
    phi_yplus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] >= 0)
    phi_yplus = phi_yplus + (phi_yplus < 0).astype(np.int) * (np.pi)
    phi_yminus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] < 0)
    phi_yminus = phi_yminus + (phi_yminus > 0).astype(np.int) * (-np.pi)
    spherical_pt[:,2] = phi_yminus + phi_yplus

    # spherical_pt[:,2] = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) 
    # spherical_pt[:,2] = spherical_pt[:,2] + (spherical_pt[:,2] > 0).astype(np.int) * (-np.pi)

    return spherical_pt

# def spherical2cartesian(pt):
#     cartesian_pt = np.zeros(pt.shape)
#     cartesian_pt[:,0] = pt[:,0]*np.sin(pt[:,1]) * np.cos(pt[:,2])
#     cartesian_pt[:,1] = pt[:,0]*np.sin(pt[:,1]) * np.sin(pt[:,2])
#     cartesian_pt[:,2] = pt[:,0]*np.cos(pt[:,1])

#     return cartesian_pt
    # return np.concatenate(((pt[:,0]*np.sin(pt[:,1]) * np.cos(pt[:,2])).reshape(-1, 1), (pt[:,0]*np.sin(pt[:,1]) * np.sin(pt[:,2])).reshape(-1, 1), (pt[:,0]*np.cos(pt[:,1])).reshape(-1, 1)), axis = 1)

def spherical2cartesian(pt):
    cartesian_pt = torch.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0]*torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0]*torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0]*torch.cos(pt[:,1])

    # cartesian_pt = np.zeros(pt.shape)
    # cartesian_pt[:,0] = pt[:,0]*np.sin(pt[:,1]) * np.cos(pt[:,2])
    # cartesian_pt[:,1] = pt[:,0]*np.sin(pt[:,1]) * np.sin(pt[:,2])
    # cartesian_pt[:,2] = pt[:,0]*np.cos(pt[:,1])

    return cartesian_pt

def threshold_bin(nlos_data):
    data_sum = torch.sum(torch.sum(nlos_data, dim = 2),dim = 1)
    for i in range(0, 800, 10):
        
        # print(i)
        # print(value)
        if (data_sum[i] < 1e-12) & (data_sum[i+10] > 1e-12):
            break 
        
    threshold_former_bin = i - 10
    if threshold_former_bin > 650:
        error('error: threshold too large')
    return threshold_former_bin

# if __name__=='__main__': # test for encoding
#     pt = torch.rand(3)
#     coded_pt = encoding(pt, 10)
#     pass

# if __name__ == "__main__": # test for cartesian2spherical
#     x = np.array([1,0,0,1])
#     y = np.array([0,1,0,1])
#     z = np.array([0,0,1,1])
#     pt = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1e-4,-1e-4,1]])
#     spherical_pt = cartesian2spherical(pt)
#     print(spherical_pt)
#     pass