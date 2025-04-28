import torch
import numpy as np

def MAL(defoucs,X,Y,pSize,img,E0 = 300000,Cs = -0.01*1e7,alpha = 0.5 * 1e-03,M=5,deltaC1=1):
    pi = 3.1415926535
    m = 9.109383*1e-31
    e = 1.602177*1e-19
    c = 299792458
    h = 6.62607*1e-34
    # Cs = -0.01*1e7 #the third-order spherical aberration in A
    # alpha = 0.5 * 1e-03 #the divergence half angle of the electron beam in rad
    # E0 = 300000 #microscope voltage in V
    l = h/np.sqrt(2*m*e*E0)/np.sqrt(1 + e*E0/2/m/(c**2)) * 10**10 #wavelength in A
    deltaepsilon = 0.92*deltaC1 
    i1 = 0
    highth = X
    width = Y
    q,qx,qy = makeFourierCoords(X,Y,pSize)
    Tdm_r = torch.zeros(highth,width,2*M+1)
    Tdm_i = torch.zeros(highth,width,2*M+1)
    fm = torch.zeros(highth,width,2*M+1)
    for i in range(-M,M):
        dxu = ((defoucs + i*deltaepsilon)*l+Cs*(l**3)*((torch.abs(q))**2))*q
        Es = torch.exp(-(pi*alpha/l)**2 * dxu**2)
        P = torch.exp(-1j*pi*(l*(q**2)*(defoucs+i*deltaepsilon )+ 0.5*(l**3)*(q**4)*Cs))
        m = P * Es
        Tdm_r[:,:,i1] = m.real
        Tdm_i[:,:,i1] = m.imag
        fm[:,:,i1] = deltaepsilon /(np.sqrt(2*pi)*deltaC1) * np.exp(-(i*deltaepsilon)**(2) /(2*deltaC1**2))
        i1 += 1
    img_bf = torch.zeros(highth,width)
    for i in range(2*M+1):
        g = torch.abs(torch.fft.ifft2(torch.fft.fft2(img)*(Tdm_r[:,:,i]+1j*Tdm_i[:,:,i])))**2
        img_bf += fm[:,:,i] * g
    return img_bf





def MAL_Astigmatism(defoucs,X,Y,pSize,img,E0 = 300000,Cs = -0.01*1e7,alpha = 0.5 * 1e-03,M=5,deltaC1=1,theta=0,Z = 150):
    pi = 3.1415926535
    m = 9.109383*1e-31
    e = 1.602177*1e-19
    c = 299792458
    h = 6.62607*1e-34
    # Cs = -0.01*1e7 #the third-order spherical aberration in A
    # alpha = 0.5 * 1e-03 #the divergence half angle of the electron beam in rad
    # E0 = 300000 #microscope voltage in V
    l = h/np.sqrt(2*m*e*E0)/np.sqrt(1 + e*E0/2/m/(c**2)) * 10**10 #wavelength in A
    deltaepsilon = 0.92*deltaC1 
    i1 = 0
    highth = X
    width = Y
    # theta = 0 
    # Z = 150 #A
    q,qx,qy = makeFourierCoords(X,Y,pSize)
    Tdm_r = torch.zeros(highth,width,2*M+1)
    Tdm_i = torch.zeros(highth,width,2*M+1)
    fm = torch.zeros(highth,width,2*M+1)
    for i in range(-M,M):
        dxu = ((defoucs + i*deltaepsilon)*l+Cs*(l**3)*((torch.abs(q))**2))*q
        Es = torch.exp(-(pi*alpha/l)**2 * dxu**2)
        P = torch.exp(-1j*pi*(l*(q**2)*(defoucs+i*deltaepsilon )+ 0.5*(l**3)*(q**4)*Cs + Z/2*(q**2)*l*torch.sin(2*(torch.angle(qx + 1j*qy)- pi/4-theta))))
        m = P * Es
        Tdm_r[:,:,i1] = m.real
        Tdm_i[:,:,i1] = m.imag
        fm[:,:,i1] = deltaepsilon /(np.sqrt(2*pi)*deltaC1) * np.exp(-(i*deltaepsilon)**(2) /(2*deltaC1**2))
        i1 += 1
    img_bf = torch.zeros(highth,width)
    for i in range(2*M+1):
        g = torch.abs(torch.fft.ifft2(torch.fft.fft2(img)*(Tdm_r[:,:,i]+1j*Tdm_i[:,:,i])))**2
        img_bf += fm[:,:,i] * g
    return img_bf

def normalize(tensor):
    min_val = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def defocusfitting(defoucs,real_image,X,Y,rangex1,rangex2,rangey1,rangey2,pSize,img,E0 = 300000,Cs = -0.01*1e7,alpha = 0.5 * 1e-03,M=5,deltaC1=1):
    pi = 3.1415926535
    m = 9.109383*1e-31
    e = 1.602177*1e-19
    c = 299792458
    h = 6.62607*1e-34
    l = h/np.sqrt(2*m*e*E0)/np.sqrt(1 + e*E0/2/m/(c**2)) * 10**10 
    # deltaepsilon = 0.4*deltaC1 
    deltaepsilon = 5*deltaC1 
    i1 = 0
    highth = X
    width = Y
    q,qx,qy = makeFourierCoords(X,Y,pSize)
    Tdm_r = torch.zeros(highth,width,2*M+1)
    Tdm_i = torch.zeros(highth,width,2*M+1)
    fm = torch.zeros(highth,width,2*M+1)
    for i in range(-M,M):
        dxu = ((defoucs + i*deltaepsilon)*l+Cs*(l**3)*((torch.abs(q))**2))*q
        Es = torch.exp(-(pi*alpha/l)**2 * dxu**2)
        P = torch.exp(-1j*pi*(l*(q**2)*(defoucs+i*deltaepsilon )+ 0.5*(l**3)*(q**4)*Cs))
        m = P * Es
        Tdm_r[:,:,i1] = m.real
        Tdm_i[:,:,i1] = m.imag
        i1 += 1
    score = {}
    for i in range(2*M+1):
        g = torch.abs(torch.fft.ifft2(torch.fft.fft2(img)*(Tdm_r[:,:,i]+1j*Tdm_i[:,:,i])))**2
        score[float(defoucs + (i-5)*deltaepsilon)] = torch.mean((g[rangex1:rangex2,rangey1:rangey2]-real_image).pow(2)).sum()
    new_defocus = min(zip(score.values(),score.keys()))[1]
    return new_defocus

def makeFourierCoords(X,Y,pSize):
    
    if X % 2 == 0:
        tempX = torch.cat((torch.linspace(0, int(X/2), int(X/2+1)) , torch.linspace(int(-X/2+1), -1, int(X/2-1))), 0)
        tempX = tempX/(X*pSize)
    else:    
        tempX = torch.cat((torch.linspace(0, int(X/2-0.5), int(X/2-0.5)) , torch.linspace(int(-X/2-0.5), -1, int(X/2-0.5))), 0)
        tempX = tempX/(X*pSize)
    
    tempX = tempX.reshape([-1, 1]).repeat(1, Y)

    if Y % 2 == 0:
        tempY = torch.cat((torch.linspace(0, int(Y/2), int(Y/2+1)) , torch.linspace(int(-Y/2+1), -1, int(Y/2-1))), 0)
        tempY = tempY/(Y*pSize)
    else:    
        tempY = torch.cat((torch.linspace(1, int(Y/2-0.5), int(Y/2-0.5)) , torch.linspace(int(-Y/2-0.5), -1, int(Y/2-0.5))), 0)
        tempY = tempY/(Y*pSize)
    
    tempY = tempY.reshape([1, -1]).repeat(X, 1)

    iradius = torch.sqrt(tempX.pow(2) + tempY.pow(2))
    iradius = iradius.double()
    return iradius,tempX,tempY

def makeFourierCoords1(N, pSize):
    if N % 2 == 0:
        q = torch.roll( torch.arange((-N/2),(N/2-1)+1) / (N*pSize), int(-N/2))
    else:
        q = torch.roll( torch.arange((-N/2+0.5),(N/2-0.5)+1) / ((N-1)*pSize), int(-N/2))
    return q

def MIMAP(defoucs,X,Y,pSize,img,E0 = 300000,Cs = -0.01*1e7,alpha = 0.5 * 1e-03,M=5,deltaC1=1):
    pi = 3.1415926535
    m = 9.109383*1e-31
    e = 1.602177*1e-19
    c = 299792458
    h = 6.62607*1e-34
    # Cs = -0.01*1e7 #the third-order spherical aberration in A
    # alpha = 0.5 * 1e-03 #the divergence half angle of the electron beam in rad
    # E0 = 300000 #microscope voltage in V
    # Suppose alpha = l*q0
    l = h/np.sqrt(2*m*e*E0)/np.sqrt(1 + e*E0/2/m/(c**2)) * 10**10 #wavelength in A
    deltaepsilon = 0.92*deltaC1 
    i1 = 0
    highth = X
    width = Y
    k = makeFourierCoords(X,Y,pSize)
    Tdm_r = torch.zeros(highth,width)
    Tdm_i = torch.zeros(highth,width)
    fm = torch.zeros(highth,width)
    epsilon0 = (pi*alpha*delta)
    # ReXk = 