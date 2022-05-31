import numpy as np
import random
import scipy
import scipy.io
import torch

# DRUNet borrowed from https://github.com/cszn/DPIR/tree/master/model_zoo
from external.utils_dpir import test_mode as test_mode_dpir


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



def apply_model(x_cur, model=None, nonblind=False, ths=1, arch=None):
    """
    Applies the DNN to a NumPy image of dimension (C, W, H).
    """

    imgn = torch.from_numpy(x_cur)
    init_shape = imgn.shape
    if len(init_shape) == 2:
        imgn.unsqueeze_(0)
        imgn.unsqueeze_(0)
    elif len(init_shape) == 3:
        imgn.unsqueeze_(0)
    else:
        raise ValueError("Inappropriate image dims.")
    imgn = imgn.type(Tensor)

    with torch.no_grad():
        if nonblind:
            ths_np = np.asarray([ths])
            sig = torch.from_numpy(ths_np).unsqueeze_(1).unsqueeze_(1).unsqueeze_(1).type(Tensor)
            out_net = model(imgn, sig)
        elif arch == 'DRUNet':
            ths_map = torch.FloatTensor([ths]).repeat(1, 1, imgn.shape[2], imgn.shape[3]).type(Tensor)
            img_in = torch.cat((imgn, ths_map), dim=1)  # .type(Tensor)
            if img_in.size(2) // 8 == 0 and img_in.size(3) // 8 == 0:
                out_net = model(img_in)
            else:
                out_net = test_mode_dpir(model, img_in, refield=64, mode=5)
        else:
            out_net = model(imgn)

    img = out_net[0].cpu().detach().numpy()
    if len(init_shape) == 2:
        x = img[0]
    elif len(init_shape) == 3:
        x = img

    return x

