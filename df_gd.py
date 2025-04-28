import cv2
import numpy as np
import torch
import os
# import natsort
from scipy import io
from utils.preprocess import focalSeries
from torch.fft import fft2,ifft2
from utils.multiCorr import multiCorr, makeFourierCoords1
import utils.DM as dm
import pdb
import time
from tqdm import * 
import matplotlib.pyplot as plt
from utils.CTF_yy import CTF, makeFourierCoords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # load simulated images
# path = "./data/focal_simulation2/"
# resolution_x = resolution_y = 1024
# defocus_series = torch.linspace(-200, 200, 41)
# padsize = 0
# files = os.listdir(path)
# files = natsort.natsorted(files)
# images = np.zeros([ resolution_x, resolution_y,defocus_series.shape[0]])
# for i in range(len(files)):
#     image1 = cv2.imread(path+ files[i], -1)
#     image1 = np.array(image1)
#     images[ :, :,i] = image1
# images = torch.Tensor(images)
# x = focalSeries(images,0.01726,defocus_series,E0= 300000,padsize = padsize,Cs = -0.01*1e7,alpha = 0.5 * 1e-06 , deltaC1=20)
# active = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]

# load real shot images
image, dimensions, calibration, metadata = dm.dm_load('/public/home/liuyang2022/TEM/data/FS4_Fig1.dm3')
image = image.transpose([1,2,0])
images = torch.Tensor(image)
padsize = 0
resolution_x = 1024
resolution_y = 1024
active = [i for i in range(82)]
defocus_series = torch.linspace(990, 180, 82)
x = focalSeries(images,0.181271,defocus_series,E0= 80000,padsize = padsize,Cs = -5.76*1e4,alpha = 0.1 * 1e-03 , deltaC1=20)


# Parameter initialization
stackAmplitude = x.stackAmplitude(sigmapad=8,edgeblend = 16,qInfolimit=1,qpower=8)
stackMask1 = torch.zeros([resolution_x +  padsize, resolution_y + padsize, len(defocus_series)])
stackMask = x.stackMask()
for i in range(len(defocus_series)):
    stackMask1[:,:,i] = torch.from_numpy(stackMask)
phaseEnvelop = x.phaseEnvelop().cpu().numpy()
EWtemp,angle = x.Init(active)
ctfs_real = x.ctf_real
ctfs_imag = x.ctf_imag
prop = ctfs_real + 1j * ctfs_imag
backprop = ctfs_real -1j * ctfs_imag
EWphase = ifft2(fft2(angle) * phaseEnvelop )
recon = torch.abs(EWtemp) * torch.exp(1j*EWphase)


ABSDIFF = []
x.qx = makeFourierCoords1(resolution_x +  padsize,x.pixelsize)
x.qy = makeFourierCoords1(resolution_y +  padsize,x.pixelsize)
[x.qxa,x.qya] = torch.tensor(np.meshgrid(x.qx,x.qy, indexing='ij'))
correlationMethod = 'hybrid' # NOTE
correlationAlign = 20 # NOTE
stackSize = [x.highth + padsize, x.width + padsize, x.page]


x.dxy = torch.zeros(stackSize[2], 2)
# Aligned data by Mactempas
# x.dxy[:,0] = torch.Tensor([2.68318915164683,2.71604221233877,4.20468628759253,3.60880726225544,3.25760802549428,2.14444287243826,1.83332095199670,-0.604151581036935,-0.215276909659924,-1.31658001269967,-1.97418756669560,-1.88920409399364,-0.805921994264259,2.41951670860045,0.682630568758355,1.02154431170449,10.1303898736751,3.96370529728498,3.64034977322961,4.43432925955645,7.98909426108656])
# x.dxy[:,1] = torch.Tensor([-4.83267915113812,-5.78448585035308,-6.32641070222663,-7.21603459825518,-6.45068639802048,-6.87717442686942,-8.88188969892341,-11.5204612440563,-13.9090029629487,-14.8721537390737,-13.1322153785011,-13.1839899886953,-10.8128127365342,-11.3090088336271,-13.7682683622223,-12.3852232851271,22.2022600610324,24.9000973097218,-13.9728528543959,-15.4258130961060,-22.4903122902559])

x.dxy[:,0] = torch.Tensor([9.27376292172652,9.87848974824416,
9.13899132068252,8.00883133227789,8.05527515571979,6.06714603453367,5.32859380374076,4.92760947427330,4.99801262211890,5.30630826784289,
6.02997503133925,6.27597108338224,5.50761963086897,5.41965149693328,6.24843599612101,7.02386104307043,6.83561396724366,7.54790805425431,
7.76137092661032,7.64164129408421,8.23882790365639,7.78198242158268,7.53453916813365,7.28171796991293,8.41004746044247,7.52235821608547,
7.33036023038865,6.86872120936599,6.06028998424979,6.33184489478938,6.40292058146723,6.19691207791938,6.57015502562321,6.43358488261649,
7.16032939736038,7.71860839390916,8.16588529691573,8.55987573863421,7.74081981656727,7.42913842605772,7.56750633974789,7.82420883243556,
7.49775951625690,7.76454564149361,7.64616011543459,7.45224146159939,6.82375642290920,6.26167379280092,6.88671003980280,8.17790317062357,
8.16026616338488,8.59075592479019,8.61011458418346,8.73610557982142,8.55103562364779,8.03257648043848,7.82258059102761,8.23991903192015,
8.20434840466200,8.25020251406092,8.24944582917127,7.93209313051118,8.02365424277870,7.92750121653736,7.18712938483937,7.12448987998182,
7.62433553256505,8.48415155002145,8.57859728436361,8.42459357767551,8.04125573497186,7.40590268553040,7.05623880654112,6.38709444203876,
6.50620840909281,9.58293538774331,8.81608210838321,8.37243764387265,7.57679447533600,7.02049258411522,25.8480255128381,9.79764774486576])
x.dxy[:,1] = torch.Tensor([-2.42783054810834,-2.83484352546491,
-5.35822131530562,-5.07518273219534,-10.1035533371901,-9.03598171480737,-9.40705534374537,-9.52429318758757,-9.45053767689574,-9.92095237223765,
-9.30298585004938,-8.73287745267353,-8.70255447116143,-8.90399088802675,-8.51312489166021,-9.09072179084729,-9.10943422326178,-9.02088985948761,
-7.91439009106966,-7.66536384604481,-6.45958874466930,-6.96006995977342,-6.95162702265487,-6.64229201329884,-6.09989801726650,-5.45974828300868,
-4.93638206367471,-4.22690722372312,-4.52365706920193,-4.42898175351848,-3.85000137597890,-3.93110332405581,-4.74921477064731,-4.59017130370453,
-4.08535955400852,-2.74219689347338,-2.73858721072193,-2.18973930485831,-1.55421038747949,-1.03448892131243,-0.825936455064729,-0.120332542795277,
0.456387154491361,1.20691770725533,2.25452370878419,2.63236056645026,2.84061426742281,2.68619792751580,2.80920752829817,5.95890746829440,
6.37723806200552,6.34612577654745,6.28896462558019,6.24404223699833,7.04606112624450,7.27537915844155,8.06338863856125,8.64992421055347,
9.65391727649335,9.94130374044507,10.3474085424328,10.4665815101705,10.9364560457804,12.2277287925418,12.7780988148448,12.4351014703898,
13.4528783089047,13.7809788836699,14.5801187800522,15.1050558333051,15.8623921510057,15.8266762983336,15.4192374352770,15.1610809144036,
14.9091511634121,14.7143510355991,14.2169328885503,15.5604918170696,15.5224569653476,15.4434965566969,-4.18430680734571,14.2938887402074])


# Apply the alignments
images1 = torch.zeros_like(images)
for a0 in range(0, stackSize[2]):
    psiShift =  torch.exp(-2j*np.pi*x.pixelsize*(x.qxa*x.dxy[a0,0] + x.qya*x.dxy[a0,1]))
    stackAmplitude[:,:,a0] = ifft2(fft2(stackAmplitude[:,:,a0]) * psiShift)
    stackMask1[:,:,a0] = ifft2(fft2(stackMask1[:,:,a0]) * psiShift)
    # images1[:,:,a0] = ifft2(fft2(images[:,:,a0]) * psiShift)


# ########

# imageAlign = torch.mean(stackAmplitude, 2)
# minShift =0.01
# PD= []
# flagAlign = False
# x.alignIterStep = 2 # NOTE

# #Main loop
# for iter in tqdm(range(20)):    
#     stackEW_real = torch.zeros([resolution_x +  padsize, resolution_y + padsize, len(active)])
#     stackEW_imag = torch.zeros([resolution_x +  padsize, resolution_y + padsize, len(active)])
#     sigAbsDiff = 0
#     for i1 in active:

#         #forward propagate
#         stackEW_real[:,:,i1] = ifft2(fft2(recon) * prop[:,:,i1]).real
#         stackEW_imag[:,:,i1] = ifft2(fft2(recon) * prop[:,:,i1]).imag
        
#         #calculate difference
#         abs = torch.abs(stackEW_real[:,:,i1]+1j*stackEW_imag[:,:,i1])
#         sigAbsDiff += np.mean(np.abs((stackAmplitude[:,:,i1] - abs).cpu().numpy()))
        
#         #update amplitude
#         EW = ( stackAmplitude[:,:,i1] * stackMask1[:,:,i1] + abs * (1-stackMask1[:,:,i1]) ) * torch.exp(1j * torch.angle(stackEW_real[:,:,i1]+1j*stackEW_imag[:,:,i1])) 
#         stackEW_real[:,:,i1] = EW.real
#         stackEW_imag[:,:,i1] = EW.imag
#     ABSDIFF.append(sigAbsDiff)

#     #back propagate
#     sigAbsDiff = sigAbsDiff / len(active)
#     fft2real = torch.zeros([resolution_x +  padsize, resolution_y + padsize,len(active)])
#     fft2imag = torch.zeros([resolution_x +  padsize, resolution_y + padsize,len(active)])
#     i2 = 0
#     for i in active:       
#         result = fft2((stackEW_real[:,:,i]+1j*stackEW_imag[:,:,i])) * backprop[:,:,i]
#         fft2real[:,:,i] = result.real
#         fft2imag[:,:,i] = result.imag
#     EWtemp = ifft2(torch.mean((fft2real +1j*fft2imag),2)) 
#     phaseangle = torch.angle(EWtemp)
#     EWphase = ifft2(fft2(phaseangle) * phaseEnvelop)
#     phaseDiff = torch.mean(torch.abs(torch.angle(recon)-EWphase))
#     PD.append(phaseDiff)
#     recon = torch.abs(EWtemp) * torch.exp(1j*EWphase)
#     flagAlign = True

# # Output
# io.savemat(f'recon_ours_{correlationMethod}.mat',{'recon':recon.cpu().numpy()})


# # ##########
recon = io.loadmat("/public/home/liuyang2022/TEM/recon_ours_hybrid.mat")
recon = torch.tensor(recon["recon"])


class df(torch.nn.Module):
    def __init__(self,num_o,defocus_series):
        super(df,self).__init__()
        self.linear = torch.nn.Linear(1,num_o,False)
        self.linear.weight = torch.nn.Parameter(defocus_series,requires_grad=True)
    def forward(self,x):
        x1 = self.linear(x)
        return x1
    
num = defocus_series.shape[0]
defocus_series = defocus_series.reshape([num,1])

model = df(num,defocus_series)
model = torch.nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=100000)
criterion = torch.nn.MSELoss(reduction="mean")

recon = recon.repeat(num,1,1).cuda()

for a0 in range(0, stackSize[2]):
    a = stackAmplitude[:,:,a0] 
    mean = torch.sqrt(torch.mean(a**2))  
    stackAmplitude[:,:,a0] = (a/mean)**2
images1 = stackAmplitude.permute(2,0,1).cuda()
cm = makeFourierCoords(x.highth+padsize,x.width+padsize,x.pixelsize).cuda()
for ii in trange(100):

    # for name, parms in model.named_parameters():	
    #             print('-->name:', name)
    #             print('-->para:', parms)
    #             print('-->grad_requirs:',parms.requires_grad)
    #             print('-->grad_value:',parms.grad)
    #             print("===")

    defocus_series = model(torch.tensor([1],dtype=torch.float))

    ctfs =torch.complex(torch.ones([x.highth+padsize,x.width+padsize,num]),torch.ones([x.highth+padsize,x.width+padsize,num])).cuda()
    for i in range(defocus_series.shape[0]):
        m = (CTF(defocus_series[i], cm,E0 = 300000,Cs = -0.01*1e7,alpha = 0.5 * 1e-06 , deltaC1=20))
        ctfs[:,:,i] = m.cuda()
    ctfs = ctfs.permute(2,0,1)
    img = torch.fft.fft2(recon)
    img = img * ctfs
    img = torch.fft.ifft2(img)
    img = torch.abs(img) ** 2

    optimizer.zero_grad()
    loss = criterion(img, images1)
    loss.backward()
    optimizer.step()
torch.save(model, 'df.pt')
defocus_series = model(torch.tensor([1],dtype=torch.float))
print(defocus_series)
