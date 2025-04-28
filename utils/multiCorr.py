import numpy as np
import math
import scipy.io

def multiCorr(G1, G2, *varargin):
    xyShift = None
    G2shift = None
    imageCorrIFT = None
    ###
    methodCorr, upsampleFactor, peakSearchMode, peakList = ParseVarargin(varargin)
    ###
    # print(methodCorr, upsampleFactor, peakSearchMode, peakList)
    if G1.shape != G2.shape:
        exit('Input images are not the same dimensions!')
    
    G12 = G1 * np.conj(G2)
    
    np.clongdouble
    if methodCorr == 'phase':
        imageCorr = np.exp(1j * np.angle(G12))
    elif methodCorr == 'cross':
        imageCorr = G12
    elif methodCorr == 'hybrid':
        imageCorr = np.sqrt(np.abs(G12))*np.exp(1j*np.angle(G12))

    imageSize = imageCorr.shape
    
    if peakSearchMode == False:
        imageCorrIFT = np.real(np.fft.ifft2(imageCorr))
        val = np.max(imageCorrIFT)
        xyShift = np.zeros(3)
        xyShift[0], xyShift[1] = np.unravel_index(np.argmax(imageCorrIFT), imageCorrIFT.shape)
        xyShift[0] = xyShift[0] + 1
        xyShift[1] = xyShift[1] + 1 

        if upsampleFactor == 1:
            xyShift[0] = (xyShift[0]-1+imageSize[0]/2) % imageSize[0] - imageSize[0]/2
            xyShift[1] = (xyShift[1]-1+imageSize[1]/2) % imageSize[1] - imageSize[1]/2
            xyShift[2] = val
            G2shift = np.fft.fft2(np.roll(np.fft.ifft2(G2),xyShift[0:2]))
        else:
            imageCorrLarge = upsampleFFT(imageCorr, 2)
            imageSizeLarge = imageCorrLarge.shape
            val = np.max(imageCorrLarge)
            xyShift[0], xyShift[1] = np.unravel_index(np.argmax(imageCorrLarge), imageCorrLarge.shape)
            xyShift[0] = xyShift[0] + 1
            xyShift[1] = xyShift[1] + 1 
            xyShift[0] = (xyShift[0]-1+imageSizeLarge[0]/2) % imageSizeLarge[0] -imageSizeLarge[0]/2
            xyShift[1] = (xyShift[1]-1+imageSizeLarge[1]/2) % imageSizeLarge[1] -imageSizeLarge[1]/2
            xyShift[0:2] = xyShift[0:2]/2

            if upsampleFactor > 2:
                xyShift[0] = round(xyShift[0]*upsampleFactor)/upsampleFactor
                xyShift[1] = round(xyShift[1]*upsampleFactor)/upsampleFactor
                globalShift = np.fix(np.ceil(upsampleFactor*1.5)/2)

                imageCorrUpsample = np.conj(dftUpsample(np.conj(imageCorr),upsampleFactor,globalShift-xyShift*upsampleFactor)) / (np.fix(imageSizeLarge[0])*np.fix(imageSizeLarge[1]*upsampleFactor**2))
                val = np.max(imageCorrUpsample)
                xySubShift = np.zeros(3)
                xySubShift[0], xySubShift[1] = np.unravel_index(np.argmax(imageCorrUpsample), imageCorrUpsample.shape)
                xySubShift[0] += 1
                xySubShift[1] += 1
                try:
                    x_min = (xySubShift[0] - 1).astype(int)
                    x_max = (xySubShift[0] + 1).astype(int)
                    y_min = (xySubShift[1] - 1).astype(int)
                    y_max = (xySubShift[1] + 1).astype(int)
                    icc = np.real( imageCorrUpsample[x_min-1:x_max, y_min-1:y_max])
                    # icc = scipy.io.loadmat('../EWR/icc.mat')
                    # print(icc)
                    dx = (icc[2,1]-icc[0,1]) / (4*icc[1,1]-2*icc[2,1]-2*icc[0,1])
                    dy = (icc[1,2]-icc[1,0]) / (4*icc[1,1]-2*icc[1,2]-2*icc[1,0])
                    #print(dx)
                except:
                    dx,dy = 0,0
                xySubShift = xySubShift - globalShift - 1
                xyShift[0:2] = xyShift[0:2] + (xySubShift[0:2] + np.array([dx,dy]))/upsampleFactor
            xyShift[2] = val*upsampleFactor**2
        
        qx = makeFourierCoords1(imageSize[0], 1)
        if imageSize[1] == imageSize[0]:
            qy = qx
        else:
            qy = makeFourierCoords1(imageSize[1], 1)
        G2shift = G2*(np.outer(np.exp(-2j*math.pi*qx*xyShift[0]),np.exp(-2j*math.pi*qy*xyShift[1])))
    else:
        exit('Not implemented')

    return xyShift, G2shift, imageCorrIFT

def upsampleFFT(imageInit, upsampleFactor):
    imageSize = imageInit.shape
    imageUpSize = (imageSize[0]*upsampleFactor,imageSize[1]*upsampleFactor)
    imageUpsample = np.zeros(imageUpSize, dtype=complex)

    x_range1 = np.arange(1, np.ceil(imageSize[0]/2)+1, 1)
    x_range2 = np.arange(np.ceil(1-imageSize[0]/2), 1, 1) + imageSize[0]*upsampleFactor
    y_range1 = np.arange(1, np.ceil(imageSize[1]/2)+1, 1)
    y_range2 = np.arange(np.ceil(1-imageSize[1]/2), 1, 1) + imageSize[1]*upsampleFactor

    x_range = np.concatenate((x_range1,x_range2)).astype(int)
    y_range = np.concatenate((y_range1,y_range2)).astype(int)

    # imageUpsample[x_range,y_range] = imageInit
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            imageUpsample[x_range[i]-1, y_range[j]-1] = imageInit[i,j]

    imageUpsampleReal = np.real(np.fft.ifft2(imageUpsample))
    return imageUpsampleReal

def makeFourierCoords1(N, pSize):
    if N % 2 == 0:
        q = np.roll( np.arange((-N/2),(N/2-1)+1) / (N*pSize), [0,int(-N/2)])
    else:
        q = np.roll( np.arange((-N/2+0.5),(N/2-0.5)+1) / ((N-1)*pSize), [0,int(-N/2)])
    return q

def ParseVarargin(Varargin):
    methodCorr = 'phase'
    upsampleFactor = 1
    peakSearchMode = False
    peakList = []

    for item in Varargin:
        if type(item) is str:
            methodCorrUser = item
            if (methodCorrUser != 'phase') and (methodCorrUser != 'cross') and (methodCorrUser != 'hybrid'):
                print('Method {} is not recognized, setting method to "phase"'.format(methodCorrUser))
                methodCorr = 'phase'
            else:
                methodCorr = methodCorrUser
        else:
            if  type(item) is not list:
                upsampleFactor = round(item)
                if upsampleFactor < 1:
                    print('Upsample factor is <1, setting to 1 ...')
                    upsampleFactor = 1
            # elif len(item) == 3:
            #     peakList = item
            #     peakSearchMode = True
            else:
                print('Input variable {} is not recognized'.format(item))

    return methodCorr, upsampleFactor, peakSearchMode, peakList
 
def dftUpsample(imageCorr, upsampleFactor, xyShift):
    imageUpsampled = None
    imageSize = imageCorr.shape
    pixelRadius = 1.5
    numRow = np.ceil(pixelRadius*upsampleFactor)
    numCol = np.ceil(pixelRadius*upsampleFactor)
    a = np.arange(0, imageSize[1], 1)
    b = np.arange(0, numCol, 1)
    c = np.arange(0, numRow, 1)
    d = np.arange(0, imageSize[0], 1)
    col_1 = -1j*2*math.pi/(imageSize[1]*upsampleFactor)
    col_2 = np.fft.ifftshift(a) - np.floor(imageSize[1]/2)
    col_3 = b - xyShift[1]
    colKern = np.exp(col_1*np.outer(col_2,col_3))
    row_1 = -1j*2*math.pi/(imageSize[0]*upsampleFactor)
    row_2 = c - xyShift[0]
    row_3 = np.fft.ifftshift(d) - np.floor(imageSize[0]/2)
    rowKern = np.exp(row_1*np.outer(row_2,row_3))
    imageUpsampled = np.real(rowKern@imageCorr@colKern)
    return imageUpsampled


if __name__=="__main__":
    G1 = scipy.io.loadmat('../EWR/G1.mat')
    G1 = G1['G1']
    G2 = scipy.io.loadmat('../EWR/G2.mat')
    G2 = G2['G2']
    xyShift, G2shift, imageCorrIFT = multiCorr(G1, G2, 'hybrid',20)
    print(xyShift)