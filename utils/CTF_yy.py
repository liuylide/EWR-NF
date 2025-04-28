import torch
import numpy as np
from scipy import io


def CTF(defoucs,q,E0 = 300000,Cs = -0.01*1e7,alpha = 0.5 * 1e-03 , deltaC1=20):

    pi = 3.1415926535
    m = 9.109383*1e-31
    e = 1.602177*1e-19
    c = 299792458
    h = 6.62607*1e-34
    # Cs = -0.01*1e7 #the third-order spherical aberration in A, 1: -0.008*1e7 , 2: -0.01*1e7
    # alpha = 0.5 * 1e-06 #the divergence half angle of the electron beam in rad, 1: 0.1 * 1e-06 , 2: 0.5 * 1e-06
    # E0 = 300000 #microscope voltage in V
    l = h/np.sqrt(2*m*e*E0)/np.sqrt(1 + e*E0/2/m/(c**2)) * 10**10 #wavelength in A
    theta=l*q
    C1 = defoucs #defocus in A

    Ta = 1 
    # aperture function

    Tc = torch.exp(-1j*(pi/l*((theta**2)*C1 + 0.5*(theta**4)*Cs)))
    # lens phase shift

    # Es = torch.exp(-pi**2/(l**2)*(alpha**2)*((theta**2)*(C1**2) + 2*(theta**4)*C1*Cs + (theta**6)*(Cs**2)))
    Es = torch.exp(-pi**2/(l**2)*(alpha**2)*(Cs*(theta**3)+C1*theta)**2)
    # partial spacial

    # deltaEmrs = deltaE/(2*torch.sqrt(2*np.log(2)))
    # deltaC1 = Cc*torch.sqrt((deltaEmrs/E0)**2)
    # deltaC1=20 #focus spread of the microscope in A
    Et = torch.exp(-0.5*(pi**2)/(l**2)*(deltaC1**2)*(theta**4))
    # partial temporal

    T = Ta * Tc * Es *Et#CTF
    #T = torch.exp(-1j*pi/l*((theta**2)*C1)) *Es *Et#for test
    return T


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
    return iradius

