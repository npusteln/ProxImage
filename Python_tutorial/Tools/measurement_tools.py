
import numpy as np

import scipy
import scipy.io

from numba import jit
import pylops


#* ************************************************************************
#* Linear Convolution operators 
#* For Grayscale and RGD images
# source https://matthieutrs.github.io/
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
#* Define forward/backward operators

def get_operators(type_op='circular_deconvolution', pth_kernel='blur_models/blur_1.mat'):
    '''
    Returns the forward measurement operator and the backward (adjoint) operator
    '''

    if 'deconvolution' in type_op:
        h = scipy.io.loadmat(pth_kernel)
        h = np.array(h['blur'])
        Forward_op = lambda x: Forward_conv(x,h) 
        Backward_op = lambda x: Backward_conv(x,h)
        return Forward_op, Backward_op

    else:
        raise ValueError('Unknown operator type!')


#* ************************************************************************
#* Tomography operators 
#* For Grayscale images

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