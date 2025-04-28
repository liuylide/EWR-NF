import torch
import numpy as np
from scipy import signal
from torch.fft import fft2, fftshift,ifft2
import torch.nn.functional as F
import cv2
import math
from utils.CTF_yy import CTF, makeFourierCoords
from utils.multiCorr import multiCorr, makeFourierCoords1

class focalSeries:
    def __init__(self,stack,pixelsize,defocus_series1,ifaligned,padsize =100,E0 = 300000,Cs = -0.01*1e7,alpha = 0.5 * 1e-03 , deltaC1=1):
        # global defocus_series
        self.pi = 3.1415926535
        m = 9.109383*1e-31
        e = 1.602177*1e-19
        c = 299792458
        h = 6.62607*1e-34
        self.E0 = E0 #microscope voltage in V
        self.Cs = Cs
        self.alpha = alpha
        self.deltaC1 = deltaC1
        self.l = h/np.sqrt(2*m*e*E0)/np.sqrt(1 + e*E0/2/m/(c**2)) * 10**10 #wavelength in A
        self.pixelsize = pixelsize
        self.defocus_series = defocus_series1
        self.stack = stack
        self.highth, self.width, self.page = stack.shape
        self.padsize = padsize
        self.ifaligned = ifaligned


    def align(self,align_x,align_y):
        resolution_x = self.highth
        resolution_y = self.width
        images1 = np.zeros([resolution_x +  padsize, resolution_y + padsize, len(self.defocus_series)])
        images2 = np.pad(image,((int(padsize/2),int(padsize/2)),(int(padsize/2),int(padsize/2)),(0,0)),'edge')

        qx = makeFourierCoords1(resolution_x +  self.padsize,self.pixelsize)    
        qy = makeFourierCoords1(resolution_y +  self.padsize,self.pixelsize)
        [qxa,qya] = np.array(np.meshgrid(qx,qy, indexing='ij'))
        stackSize = [resolution_x + padsize, resolution_y + padsize, len(self.defocus_series)] #2:19
        if self.ifaligned == False:
            dxy = np.zeros((stackSize[2], 2))
            dxy[:,0] = align_x
            dxy[:,1] = align_y

        for a0 in range(0, stackSize[2]):
            if self.ifaligned == False:
                psiShift =  np.exp(-2j*np.pi*self.pixelsize*(qxa*dxy[a0,0] + qya*dxy[a0,1]))
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


        images2 = torch.Tensor(images2)
        images1 = torch.Tensor(images1)
        return images1,images2


    def tukeywin(self,edgeblend = 16):
        if edgeblend == 0:
            r1 = r2 = 0.5
        else:
            r1 = edgeblend * 2 / self.highth
            r2 = edgeblend * 2 / self.width
        window_1 = signal.windows.tukey(self.highth,r1).reshape(self.highth,1)
        window_2 = signal.windows.tukey(self.width,r2).reshape(1,self.width)
        window_2D = window_1 * window_2
        return window_2D
    
    ## test tukeywin
    # w2 = tukeywin(256,256,16)
    # cv2.imshow/('gray_scale',w2)
    # cv2.waitKey(0)

    def stackMask(self,edgeblend = 16):
        padsize = self.padsize
        padsize = int(padsize/2)
        padded = np.pad(self.tukeywin(edgeblend), (padsize,padsize), 'constant')
        return padded
    
    def Butterworth(self,qInfolimit=1,qpower=8):
        padsize = self.padsize
        coor = makeFourierCoords(self.highth+padsize,self.width+padsize,self.pixelsize)
        output = 1/ (1+torch.pow((torch.pow(coor,2)),qpower)/(qInfolimit^(2*qpower)))
        return output
    
    def kPad(self,size_X,size_Y,sigma):
        X = size_X
        Y = size_Y
        tempX = torch.cat((torch.linspace(int(-X/2), 0, int(X/2+1)) , torch.linspace(1, int((X-1)/2), int(X/2))), 0) 
        tempX = tempX.reshape([-1, 1]).repeat(1, Y)
        tempY = torch.cat((torch.linspace(int(-Y/2), 0, int(Y/2+1)) , torch.linspace(1, int((Y-1)/2), int(Y/2))), 0)
        tempY = tempY.reshape([1, -1]).repeat(X, 1)
        h = np.exp(-1 * ((tempX.pow(2) + tempY.pow(2))/(2*sigma*sigma)).cpu().numpy())
        h[h<0] = 0
        if np.sum(h) != 0 :
            h = h/np.sum(h)
        return h

    def KNorm(self, sigmapad=8):
        padsize = self.padsize
        hsize = 2 * math.ceil(4 * sigmapad) + 1
        Ik = torch.zeros((self.highth+padsize,self.width+padsize,self.page))
        padsize = int(padsize/2)
        img1 = np.ones((self.highth,self.width))
        kPad = self.kPad(hsize,hsize,sigmapad)
        kNorm = 1 / signal.convolve(img1,kPad,'same')
        stack = self.stack / torch.mean(self.stack)
        for i in range(self.page):
            im = stack[:,:,i].cpu().numpy()
            Ik1 = signal.convolve(im,kPad,'same') * kNorm
            Ik1 = np.pad(Ik1,(padsize,padsize),'edge')
            Ik[:,:,i] = torch.tensor(Ik1)
        return Ik

    def stackAmplitude(self,sigmapad=8,edgeblend = 16,qInfolimit=1,qpower=8):
        padsize = self.padsize
        SA = torch.zeros((self.highth+padsize,self.width+padsize,self.page))
        # SA = np.zeros((self.highth+padsize,self.width+padsize,self.page))
        kNorm = self.KNorm(sigmapad)
        Butterworth = self.Butterworth(qInfolimit,qpower)
        stack = self.stack / torch.mean(self.stack)
        for i in range(self.page):
            I = stack[:,:,i].cpu().numpy()
            Ipad = kNorm[:,:,i].cpu().numpy()
            Ioutput =  Ipad * ( 1 - self.stackMask(edgeblend) ) + np.pad((I * self.tukeywin(edgeblend)), (int(padsize/2),int(padsize/2)),'constant')
            Ioutput[Ioutput<0] = 0
            # stackAmplitude = torch.tensor(np.sqrt(Ioutput))
            stackAmplitude = np.sqrt(Ioutput)
            stackAmplitude = np.fft.ifft2(np.fft.fft2(stackAmplitude) * Butterworth.cpu().numpy())
            stackAmplitude = torch.tensor(stackAmplitude)
            SA[:,:,i] = stackAmplitude.real
        return SA

    def phaseEnvelop(self):
        padsize = self.padsize
        defocusMax = abs(self.defocus_series[-1])
        coor =torch.pow(makeFourierCoords(self.highth+padsize,self.width+padsize,self.pixelsize),2)
        threshold =  1 / (2 * self.l * defocusMax)
        phaseEnvelope = torch.ones([self.highth +  padsize, self.width + padsize])
        for i in range(self.highth+padsize):
            for j in range(self.width+padsize):
                if coor[i][j] <= threshold:
                    phaseEnvelope[i][j] = torch.sin(self.pi * self.l*defocusMax * coor[i][j])
        return phaseEnvelope
    
    def Init(self,active,sigmapad=8,edgeblend=16,qInfolimit=1,qpower=8):
        padsize = self.padsize
        SA = self.stackAmplitude(sigmapad,edgeblend,qInfolimit,qpower)
        fft2real = torch.zeros([self.highth +  padsize, self.width + padsize, len(active)])
        fft2imag = torch.zeros([self.highth +  padsize, self.width + padsize, len(active)])
        sM = torch.tensor(self.stackMask(edgeblend))
        ctfs_real = torch.zeros([self.highth + self.padsize, self.width + self.padsize, self.defocus_series.shape[0]])
        ctfs_imag = torch.zeros([self.highth + self.padsize, self.width + self.padsize, self.defocus_series.shape[0]])
        for i in range(self.defocus_series.shape[0]):
            m = (CTF(self.defocus_series[i], makeFourierCoords(self.highth+padsize,self.width+padsize,self.pixelsize),E0,Cs,alpha,deltaC1))
            ctfs_real[:, :, i] = m.real
            ctfs_imag[:, :, i] = m.imag
        self.ctf_real = ctfs_real
        self.ctf_imag = ctfs_imag
        for i in active:
            # result = SA[:,:,i] * sM + (1 - sM)
            result = fft2(SA[:,:,i] * sM + (1 - sM)) * (self.ctf_real[:,:,i] - 1j * self.ctf_imag[:,:,i])
            fft2real[:,:,i] = result.real
            fft2imag[:,:,i] = result.imag
        EWtemp = ifft2(torch.mean((fft2real + 1j*fft2imag),2)) 
        # EWtemp = fft2real
        phaseangle = torch.angle(EWtemp)
        return EWtemp,phaseangle




# for test
if __name__=='__main__':
    import os
    # import natsort
    from scipy import io
    import scipy
    import utils.DM as dm
    
    image, dimensions, calibration, metadata = dm.dm_load('/public/home/liuyang2022/TEM/data/Stack Image172-191_19nm-0nm.dm3')
    image = image.transpose([1,2,0])
    image = scipy.signal.wiener(image,[3,3,1])
    images = torch.Tensor(image)

    padsize = 100
    resolution_x = 1024
    resolution_y = 1024
    active = [i for i in range(20)]
    defocus_series = torch.tensor([185,176,166,163,147,136,128,117,108,100,90,80,68,60,51,41,33,25,18,12])
    x = focalSeries(images,0.1813,defocus_series,E0= 80000,padsize = padsize,Cs = -1.16*1e5,alpha = 0.1 * 1e-03 , deltaC1=20)
    stackAmplitude = x.stackAmplitude(sigmapad=8,edgeblend = 16,qInfolimit=1,qpower=8)
    stackMask1 = torch.zeros([resolution_x +  padsize, resolution_y + padsize, len(defocus_series)])
    images1 = torch.zeros([resolution_x +  padsize, resolution_y + padsize, len(defocus_series)])
    stackMask = x.stackMask()
    phaseEnvelop = x.phaseEnvelop()
    EWtemp,angle = x.Init(active)
    EWphase = ifft2(fft2(angle) * phaseEnvelop )
    io.savemat('/public/home/liuyang2022/TEM/result/EWPhase.mat',{'EP':EWphase.cpu().numpy() })
    io.savemat('/public/home/liuyang2022/TEM/result/angle.mat',{'angle':angle.cpu().numpy() })
