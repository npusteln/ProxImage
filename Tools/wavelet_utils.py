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




def coef2vec(coef, Nx, Ny):
    """
    Convert wavelet coefficients to an array-type vector, inverse operation of vec2coef.
    The initial wavelet coefficients are stocked in a list as follows:
        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
    and each element is a 2D array.
    After the conversion, the returned vector is as follows:
    [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ...,cH1.flatten(), cV1.flatten(), cD1.flatten()].
    """
    vec = []
    bookkeeping = []
    for ele in coef:
        if type(ele) == tuple:
            bookkeeping.append((np.shape(ele[0])))
            for wavcoef in ele:
                vec = np.append(vec, wavcoef.flatten())
        else:  
            bookkeeping.append((np.shape(ele)))
            vec = np.append(vec, ele.flatten())
    bookkeeping.append((Nx, Ny))     
    return vec, bookkeeping

def vec2coef(vec, bookkeeping):
    """
    Convert an array-type vector to wavelet coefficients, inverse operation of coef2vec.
    The initial vector is stocked in a 1D array as follows:
    [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ..., cH1.flatten(), cV1.flatten(), cD1.flatten()].
    After the conversion, the returned wavelet coefficient is in the form of the list as follows:
        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
    and each element is a 2D array. This list can be passed as the argument in pywt.waverec2.
    """
    ind = bookkeeping[0][0] * bookkeeping[0][1] 
    coef = [np.reshape(vec[:ind], bookkeeping[0])]
    for ele in bookkeeping[1:-1]:
        indnext = ele[0] * ele[1]
        coef.append((np.reshape(vec[ind:ind+indnext], ele), 
                     np.reshape(vec[ind+indnext:ind+2*indnext], ele), 
                     np.reshape(vec[ind+2*indnext:ind+3*indnext], ele)))
        ind += 3*indnext

    return coef

def wavedec_asarray(im, wv='db8',level=3):
    wd = pywt.wavedec2(im,wv,level=level, mode='zero')
    wd, book = coef2vec(wd, im.shape[0], im.shape[1])
    return wd, book

def waverec_asarray(wd, book, wv='db8'):
    wc = vec2coef(wd, book)
    im = pywt.waverec2(wc,wv, mode='zero')
    return im

# c_np, book = wavedec_asarray(im,'db8',level=3)
# r = waverec_asarray(c_np, book, wv='db8')

def SARA_dict(im, level=3):
    
    c_1, b_1 = wavedec_asarray(im,'db1',level=level)
    ncoef1 = len(c_1)
    c_2, b_2 = wavedec_asarray(im,'db2',level=level)
    ncoef2 = len(c_2)
    c_3, b_3 = wavedec_asarray(im,'db3',level=level)
    ncoef3 = len(c_3)
    c_4, b_4 = wavedec_asarray(im,'db4',level=level)
    ncoef4 = len(c_4)
    c_5, b_5 = wavedec_asarray(im,'db5',level=level)
    ncoef5 = len(c_5)
    c_6, b_6 = wavedec_asarray(im,'db6',level=level)
    ncoef6 = len(c_6)
    c_7, b_7 = wavedec_asarray(im,'db7',level=level)
    ncoef7 = len(c_7)
    c_8, b_8 = wavedec_asarray(im,'db8',level=level)
    ncoef8 = len(c_8)
    _, b_9 = im.flatten(), im.shape
    #ncoef9 = len(c_9)
    
    def Psit(x):
        out = np.concatenate((wavedec_asarray(x,'db1',level=level)[0],
         wavedec_asarray(x,'db2',level=level)[0],
         wavedec_asarray(x,'db3',level=level)[0],
         wavedec_asarray(x,'db4',level=level)[0],
         wavedec_asarray(x,'db5',level=level)[0],
         wavedec_asarray(x,'db6',level=level)[0],
         wavedec_asarray(x,'db7',level=level)[0],
         wavedec_asarray(x,'db8',level=level)[0],
         x.flatten()))
        return out/np.sqrt(9)
    
    def Psi(y):
        out = waverec_asarray(y[:ncoef1], b_1, wv='db1')
        out = out+waverec_asarray(y[ncoef1:ncoef1+ncoef2], b_2, wv='db2')
        out = out+waverec_asarray(y[ncoef1+ncoef2:ncoef1+ncoef2+ncoef3], b_3, wv='db3')
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3:ncoef1+ncoef2+ncoef3+ncoef4], b_4, wv='db4')
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5], b_5, wv='db5')
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6], b_6, wv='db6')
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7], b_7, wv='db7')
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+ncoef8], b_8, wv='db8')
        out = out+np.reshape(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+ncoef8:], b_9)
        return out/np.sqrt(9)
    
    return Psit, Psi

def wavelet_op(im, wav='db8', level=3, test_ajd=False):
    
    _, b_8 = wavedec_asarray(im,wav,level=level)
    #ncoef8 = len(c_8)
    
    def Psi(x):
        out = wavedec_asarray(x,wav,level=level)[0]
        return out
    
    def Psit(y):
        out = waverec_asarray(y, b_8, wv=wav)
        return out
    
    if test_ajd is True:
        print('-----------------------------')
        print('Test wavelet')
        xtmp = np.random.rand(*im.shape)
        Ptxtmp = Psi(xtmp)
        ytmp = np.random.rand(*Ptxtmp.shape)
        Pytmp = Psit(ytmp)
        fwd = np.sum(xtmp.flatten()*Pytmp.flatten())
        bwd = np.sum(Ptxtmp.flatten()*ytmp.flatten())
        print('forward: ', fwd)
        print('backward: ', bwd)
        print('-----------------------------')

    return Psi, Psit

