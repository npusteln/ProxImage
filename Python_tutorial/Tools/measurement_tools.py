
import numpy as np

import scipy
import scipy.io

from numba import jit
import pylops


#* ************************************************************************
#* Linear Convolution operators 
#* For Grayscale and RGD images

def Forward_conv(xo, a):
    '''
    Implements the linear operator A:xo -> xo*a
    '''
    d = len(np.shape(xo))
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(A * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]):
            y[i,...] = np.real(np.fft.ifft2(A * np.fft.fft2(xo[i,...])))
    return y

def Backward_conv(xo, a):
    '''
    Implements the adjoint A' of the linear operator A:xo -> xo*a
    '''
    d = len(np.shape(xo))
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo[i, ...])))
    return y

#* ************************************************************************
#* Circular Convolution operators 
#* For Grayscale and RGD images

def Forward_conv_circ(xo, a):
    '''
    Implements the linear operator A:xo -> xo*a, but now * is the circular convolution
    '''
    d = len(np.shape(xo))
    l = a.shape[0]
    if d==2:
        xo = np.pad(xo, ((l//2+1,l//2),(l//2+1, l//2)), 'wrap')
    else:
        xo = np.pad(xo, ((0,0),(l//2+1,l//2),(l//2+1, l//2)), 'wrap')
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(A * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]): 
            y[i,...] = np.real(np.fft.ifft2(A * np.fft.fft2(xo[i,...])))
    return y[...,l:,l:]

def Backward_conv_circ(xo, a):
    '''
    Implements the adjoint A' of the linear operator A:xo -> xo*a, but now * is the circular convolution
    '''
    d = len(np.shape(xo))
    l = a.shape[0]
    if d==2:
        xo = np.pad(xo, ((l//2+1,l//2),(l//2+1, l//2)), 'wrap')
    else:
        xo = np.pad(xo, ((0,0),(l//2,l//2),(l//2, l//2)), 'wrap')
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo[i, ...])))
    return y[...,:-l+1,:-l+1]


#* ************************************************************************
#* Define forward/backward operators

def get_operators(type_op='circular_deconvolution', pth_kernel='blur_models/blur_1.mat', cross_cor=True):
    '''
    Returns the forward measurement operator and the backward (adjoint) operator
    Feel free to add your new operator class by adding an elif condition!
    '''

    if 'circular_deconvolution' in type_op:
        h = scipy.io.loadmat(pth_kernel)
        h = np.array(h['blur'])

        if cross_cor==True: # This is when the forward model contains a cross correlation and not a convolution (case of Corbineau's paper)
            h = np.flip(h,1)
            h = np.flip(h,0).copy() # Contiguous pb
        Forward_op = lambda x: Forward_conv_circ(x,h) # appropriate zero-padding
        Backward_op = lambda x: Backward_conv_circ(x,h)
        return Forward_op, Backward_op

    elif 'deconvolution' in type_op:
        h = scipy.io.loadmat(pth_kernel)
        h = np.array(h['blur'])
        Forward_op = lambda x: Forward_conv(x,h) 
        Backward_op = lambda x: Backward_conv(x,h)
        return Forward_op, Backward_op

    else:
        raise ValueError('Unknown operator type!')

def get_operators_tomo(type_op='tomography', ntheta = 20, nx = 250, ny = 250 ):
    '''
    Returns the forward measurement operator and the backward (adjoint) operator
    Feel free to add your new operator class by adding an elif condition!
    '''

    if 'tomography' in type_op:
        @jit(nopython=True)
        def radoncurve(x, r, theta):
            return (
                (r - ny // 2) / (np.sin(np.deg2rad(theta)) + 1e-15)
                + np.tan(np.deg2rad(90 - theta)) * x
                + ny // 2
            )

        theta = np.linspace(0.0, 180.0, ntheta, endpoint=False)

        RLop = pylops.signalprocessing.Radon2D(
            np.arange(ny),
            np.arange(nx),
            theta,
            kind=radoncurve,
            centeredh=True,
            interp=False,
            engine="numba",
            dtype="float64",
        )
        
        LipPhi = np.real((RLop.H*RLop).eigs(neigs=1, which='LM')[0])
        
        Forward_op = lambda x: RLop.H * x.ravel()/LipPhi # appropriate zero-padding
        Backward_op = lambda x: RLop * x.ravel()/LipPhi
        return Forward_op, Backward_op
    
    else:
        raise ValueError('Unknown operator type!')

#* ************************************************************************
#* Tomography operators 
#* For Grayscale images


# @jit(nopython=True)
# def radoncurve(x, r, theta):
#     return (
#         (r - ny // 2) / (np.sin(np.deg2rad(theta)) + 1e-15)
#         + np.tan(np.deg2rad(90 - theta)) * x
#         + ny // 2
#     )

# theta = np.linspace(0.0, 180.0, ntheta, endpoint=False)

# RLop = pylops.signalprocessing.Radon2D(
#     np.arange(ny),
#     np.arange(nx),
#     theta,
#     kind=radoncurve,
#     centeredh=True,
#     interp=False,
#     engine="numba",
#     dtype="float64",
# )

def get_operators_tomo2(type_op='tomography', ntheta = 100, nx =224 , ny =224 ):
    '''
    Returns the forward measurement operator and the backward (adjoint) operator
    Feel free to add your new operator class by adding an elif condition!
    '''

    if 'tomography' in type_op:
        @jit(nopython=True)
        def radoncurve(x, r, theta):
            return (
                (r - ny // 2) / (np.sin(np.deg2rad(theta)) + 1e-15)
                + np.tan(np.deg2rad(90 - theta)) * x
                + ny // 2
            )

        theta = np.linspace(0.0, 180.0, ntheta, endpoint=False)

        RLop = pylops.signalprocessing.Radon2D(
            np.arange(ny),
            np.arange(nx),
            theta,
            kind=radoncurve,
            centeredh=True,
            interp=False,
            engine="numba",
            dtype="float64",
        )
        
        LipPhi = np.real((RLop.H*RLop).eigs(neigs=1, which='LM')[0])
        
        Forward_op = lambda x: RLop.H * x.ravel()/np.sqrt(LipPhi) # appropriate zero-padding
        Backward_op_tmp = lambda x: RLop * x.ravel()/np.sqrt(LipPhi)
        Backward_op = lambda x:  Backward_op_tmp(x).reshape(nx,ny)       
        return Forward_op, Backward_op
    
    else:
        raise ValueError('Unknown operator type!')