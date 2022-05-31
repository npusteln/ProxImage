import time
import random

import numpy as np
from numpy import random 
import pywt
from skimage import data

import os

import scipy
import scipy.io
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from .wavelet_utils import *
from .proximal_tools import *
from .tools import *
from .measurement_tools import *


def FB_l1_linop(y, Phi, Phit, Psi, Psit, x=None, gamma=1.9, mu = 1., max_iter=1000, display_it=100, folder_save='./', save_name='test_fb', crit_norm=1e-4, x_true=None):
    '''
    FB algorithm with l1-SARA reg
    '''

    # Initialisation - save values
    data_fid = np.zeros(max_iter+1)
    reg = np.zeros(max_iter+1)
    crit = np.zeros(max_iter+1)
    snr_it = np.zeros(max_iter+1)
    norm_it = np.zeros(max_iter)
    norm_it_rel = np.zeros(max_iter)
    norm_grad = np.zeros(max_iter)
    
    # Initialisation
    if x is None:
        x = Phit(y)
    if x_true is None:
        x_true = np.copy(x)
    
    normPhi2 = op_norm2(Phi, Phit, x.shape)
    gamma = gamma/normPhi2
    normPsi2 = op_norm2(Psi, Psit, x.shape)

    Phix = Phi(x)
    res = Phix-y
    0.5 * np.sum(np.absolute(res.flatten()))**2
    Px = Psi(x)
    reg[0] = mu * np.sum( np.absolute(Px.flatten()))
    snr_it[0] = snr_numpy(x_true,x)

    u,v = None, None

    for iter in range(max_iter):
        
        x_old = np.copy(x)

        grad = Phit(res)
        x_ = x-gamma*grad 
        x,u,v = dualFB_l1_linop_pos(x_, Psit, Psi, mu*gamma, u,v, Psi_norm2=normPsi2)
        

        Phix = Phi(x)
        res = Phix-y 
        data_fid[iter+1] = 0.5 * np.sum(np.absolute(res.flatten()))**2
        Px = Psi(x)
        reg[iter+1] = mu * np.sum( np.absolute(Px.flatten()) )
        crit[iter+1] = data_fid[iter+1] + reg[iter+1]
        snr_it[iter+1] = snr_numpy(x_true,x)
        norm_it[iter] = np.linalg.norm(x.flatten()-x_old.flatten())
        norm_it_rel[iter] = norm_it[iter]/np.linalg.norm(x.flatten())
        norm_grad[iter] = np.linalg.norm(grad.flatten())
        
        if iter%display_it == 0:
            print('iter = ', iter)
            print('snr = ', snr_it[iter+1])
            print('data fidelity: ', data_fid[iter+1])
            print('crit: ', crit[iter+1])
            print('relative norm it: ', norm_it_rel[iter])
            print('      vs. stop norm: ', crit_norm)
            print('norm grad: ', norm_grad[iter])
            print('-------------------------------')



        if norm_it_rel[iter] < crit_norm:
            print('Algo stopped it: ', iter)
            data_fid = data_fid[:iter+2]
            reg = reg[:iter+2]
            crit = crit[:iter+2]
            snr_it = snr_it[:iter+2]
            norm_it = norm_it[:iter+1]
            norm_it_rel = norm_it_rel[:iter+1]
            norm_grad = norm_grad[:iter+1]
            break


    save_name_end = folder_save+save_name+'_save_final.npy'
    np.save(save_name_end, {'im_rec':x, 'snr':snr_it, 'data_fid':data_fid, 'norm_it':norm_it, 'norm_it_rel':norm_it_rel})
    
    return x, snr_it, crit, norm_it
    

