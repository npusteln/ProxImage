import numpy as np





def GradientHor(x):
    y=x-np.roll(x,1,axis=1)
    y[:,0]=0
    return y
def GradientVer(x):
    y=x-np.roll(x,1,axis=0)
    y[0,:]=0
    return y
def DivHor(x):
    N=len(x[0])
    y=x-np.roll(x,-1,axis=1)
    y[:,0]=-x[:,1]
    y[:,N-1]=x[:,N-1]
    return y
def DivVer(x):
    N=len(x)
    y=x-np.roll(x,-1,axis=0)
    y[0,:]=-x[1,:]
    y[N-1,:]=x[N-1,:]
    return y


def tv_op(im, test_ajd=False):

    def Psi(x):
        y=[]
        y.append(GradientHor(x))
        y.append(GradientVer(x))
        return np.asarray(y)
    def Psit(y):
        x=DivHor(y[0])+DivVer(y[1])
        return x
    
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
    
