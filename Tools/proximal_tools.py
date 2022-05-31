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





def soft_thresh(x,ths):
    p = np.sign(x) * np.maximum(np.absolute(x)-ths, 0)
    return p




def dualFB_l1_linop_pos(xbar, Psit, Psi, mu, u=None,v=None, Psi_norm2=1, norm_stop = 1e-4, max_iter = 500):
    '''
    Dual FB to compute the proximity operator of 
    l1 norm in a sparsifying dictionary, with positivity constraints
    ---
    xbar: starting point
    Psi:  forward sparsifying operator
    Psit: backward sparsifying operator
    Psi_norm2: (square) Spectral norm of operator Psi
    mu:   regularisation parameter
    u,v:    dual variable
    '''

    gam = 1.9/Psi_norm2
    x = xbar.copy()
    if u is None:
        u = Psi(x)
        v = x.copy()
    
    norm_it = np.zeros(max_iter+1)
    crit = np.zeros(max_iter+1)

    for iter in range(max_iter):
        xold = x.copy()

        x = xbar - 0.5*(Psit(u)+v)

        Px = Psi(x)
        u_ = u + gam * Px
        pu = soft_thresh(u_/gam, 2*mu/gam)
        u = u_ - gam * pu

        v_ = v + gam * x
        v_[v_<0]=0
        v = v_ - v_

        norm_it[iter] = np.linalg.norm(x.flatten()-xold.flatten())/np.linalg.norm(x.flatten())
        crit[iter] = np.linalg.norm(x.flatten()-xbar.flatten())**2 + mu * np.sum(np.absolute(Px))
        
        if iter>0 and iter%1000==0:
            print('iter = ', iter)
            print('norm it = ', norm_it[iter])
            print('crit = ', crit[iter])

        if iter>0 and norm_it[iter] < norm_stop:
            break

    return x, u, v


def dualFB_l1_linop(xbar, Psit, Psi, mu, u=None, Psi_norm2=1, norm_stop = 1e-4, max_iter = 700):
    '''
    Dual FB to compute the proximity operator of 
    l1 norm in a dictionary 
    ---
    xbar: starting point
    Psi:  forward sparsifying operator
    Psit: backward sparsifying operator
    Psi_norm2: (square) Spectral norm of operator Psi
    mu:   regularisation parameter
    u:    dual variable
    '''


    gam = 1.9/Psi_norm2
    x = xbar.copy()

    if u is None:
        u = Psit(x)
    
    norm_it = np.zeros(max_iter+1)
    crit = np.zeros(max_iter+1)

    for iter in range(max_iter):
        xold = x.copy()

        x = xbar - Psit(u)

        Px = Psi(x)
        u_ = u + gam * Px
        pu = soft_thresh(u_/gam, mu/gam)
        u = u_ - gam * pu


        norm_it[iter] = np.linalg.norm(x.flatten()-xold.flatten())/np.linalg.norm(x.flatten())
        crit[iter] = 0.5* np.linalg.norm(x.flatten()-xbar.flatten())**2 + mu * np.sum(np.absolute(Px))
        
        if iter>0 and iter%1000==0:
            print('iter = ', iter)
            print('norm it = ', norm_it[iter])
            print('crit = ', crit[iter])

        if iter>0 and norm_it[iter] < norm_stop:
            break

    return x, u



#* RGB prox - channel wise


def dualFB_l1_linop_posRGB(xbar, Psit, Psi, mu, u=None,v=None, norm_stop = 1e-4, max_iter = 500):
    '''
    Dual FB to compute the proximity operator of 
    l1 norm in a sparsifying dictionary, with positivity constraints
    ---
    xbar: starting point
    Psi:  forward sparsifying operator
    Psit: backward sparsifying operator
    mu:   regularisation parameter
    u,v:    dual variable
    '''


    gam = 1.9
    x = xbar.copy()

    if u is None:
        u = np.array([Psi(x[0,:,:]), Psi(x[1,:,:]), Psi(x[2,:,:])])
        v = x.copy()
    
    norm_it = np.zeros(max_iter+1)
    crit = np.zeros(max_iter+1)

    for iter in range(max_iter):
        xold = x.copy()

        Ptu = np.array([Psit(u[0]), Psit(u[1]), Psit(u[2])])
        x = xbar - 0.5*(Ptu+v)

        Px = np.array([Psi(x[0,:,:]), Psi(x[1,:,:]), Psi(x[2,:,:])])
        u_ = u + gam * Px
        pu = soft_thresh(u_/gam, 2*mu/gam)
        u = u_ - gam * pu

        v_ = v + gam * x
        v_[v_<0]=0
        v = v_ - v_

        norm_it[iter] = np.linalg.norm(x.flatten()-xold.flatten())/np.linalg.norm(x.flatten())
        crit[iter] = np.linalg.norm(x.flatten()-xbar.flatten())**2 + mu * np.sum(np.absolute(Px))
        
        if iter>0 and iter%1000==0:
            print('iter = ', iter)
            print('norm it = ', norm_it[iter])
            print('crit = ', crit[iter])

        if iter>0 and norm_it[iter] < norm_stop:
            break

    return x, u, v



def dualFB_l1_linopRGB(xbar, Psit, Psi, mu, u=None, norm_stop = 1e-4, max_iter = 700):
    '''
    Dual FB to compute the proximity operator of 
    l1 norm in a dictionary 
    ---
    xbar: starting point
    Psi:  forward sparsifying operator
    Psit: backward sparsifying operator
    mu:   regularisation parameter
    u:    dual variable
    '''


    gam = 1.9
    x = xbar.copy()

    if u is None:
        u = np.array([Psi(x[0,:,:]), Psi(x[1,:,:]), Psi(x[2,:,:])])
    
    norm_it = np.zeros(max_iter+1)
    crit = np.zeros(max_iter+1)

    for iter in range(max_iter):
        xold = x.copy()

        Ptu = np.array([Psit(u[0]), Psit(u[1]), Psit(u[2])])
        x = xbar - Ptu

        Px = np.array([Psi(x[0,:,:]), Psi(x[1,:,:]), Psi(x[2,:,:])])
        u_ = u + gam * Px
        pu = soft_thresh(u_/gam, mu/gam)
        u = u_ - gam * pu


        norm_it[iter] = np.linalg.norm(x.flatten()-xold.flatten())/np.linalg.norm(x.flatten())
        crit[iter] = 0.5* np.linalg.norm(x.flatten()-xbar.flatten())**2 + mu * np.sum(np.absolute(Px))
        
        if iter>0 and iter%1000==0:
            print('iter = ', iter)
            print('norm it = ', norm_it[iter])
            print('crit = ', crit[iter])

        if iter>0 and norm_it[iter] < norm_stop:
            break

    return x, u
