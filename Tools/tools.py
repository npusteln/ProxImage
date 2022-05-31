
import numpy as np
import matplotlib.pyplot as plt
import imageio


def snr_numpy(xtrue, x):
    snr = 20* np.log10(np.linalg.norm(xtrue.flatten())/(np.linalg.norm(xtrue.flatten()-x.flatten())+1e-6))
    return snr


def imshow(img, title=None):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()

def imshowgray(img, title=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def save_image_numpy(im, path):
    """
    Saves an image im np format
    """
    im = np.moveaxis(im, 0, -1)
    im_bounded = im*255.
    im_bounded[im_bounded>255.] = 255.
    im_bounded[im_bounded<0.] = 0.
    imageio.imwrite(path, np.uint8(im_bounded))


#* ************************************************************************
#* Power method to compute operator norm

def op_norm2(Phi, Phit, im_size):
    tol = 1e-5
    max_iter = 500
    xtmp = np.random.randn(*im_size)
    xtmp = xtmp/np.linalg.norm(xtmp.flatten())
    val = 1 
    for _ in range(max_iter):
        old_val = val
        xtmp = Phit(Phi(xtmp))
        val = np.linalg.norm(xtmp.flatten())
        rel_val = np.absolute(val-old_val)/old_val
        if rel_val < tol:
            break
        xtmp = xtmp/val
    return val