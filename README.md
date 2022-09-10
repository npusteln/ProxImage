# ProxImage
Proximal algorithms for image analysis

## Authors

#### Nelly Pustelnik, nelly.pustelnik@ens-lyon.fr 

#### Audrey Repetti, A.Repetti@hw.ac.uk

## Summary

Image processing  aims to extract or interpret the information contained in the observed data linked to one (or more) image(s). Most of the analysis tools are based on the formulation of an objective function and the development of suitable optimization methods. This class of approaches, qualified as variational, has become the state-of-the-art for many image processing modalities, thanks to their ability to deal with large-scale problems, their versatility allowing them to be adapted to different contexts, as well as the associated theoretical results ensuring convergence towards a solution of the finite objective function.

## Slides of the course

1-  Inverse problems and variational approaches - [pdf](https://github.com/npusteln/ProxImage/raw/main/Slides/Part1.pdf)

2- Variational approaches: From inverse problems to segmentation - [pdf](https://github.com/npusteln/ProxImage/raw/main/Slides/Part2.pdf)

3- Variational approaches in supervised learning - [pdf](https://github.com/npusteln/ProxImage/raw/main/Slides/Part3.pdf)

4- Optimisation algorithms - [pdf](https://github.com/npusteln/ProxImage/raw/main/Slides/Part4.pdf)

5- Optimisation algorithms: Block-coordinate approaches - [pdf](https://github.com/npusteln/ProxImage/raw/main/Slides/Part5.pdf)

6- Supervised learning for solving inverse problems - [pdf](https://github.com/npusteln/ProxImage/raw/main/Slides/Part6.pdf)



## Python notebook

1- Play with direct model - [Notebook](https://github.com/npusteln/ProxImage/blob/main/Python_tutorial/Tutorial_part1.ipynb)

2- Image deconvolution considering Forward-Backward algorithm, FISTA and Condat-Vu algorithm - [Notebook](https://github.com/npusteln/ProxImage/blob/main/Python_tutorial/Tutorial_part2.ipynb)

3- Image denoising with Plug-and-Play Forward-Backward - [Notebook](https://github.com/npusteln/ProxImage/blob/main/Python_tutorial/Tutorial_part3.ipynb)


## Required packages :

  * numpy
  
  * matplotlib
  
  * PIL
   
  * scipy
   
  * pywt
   
  * bm3d
   
  * torch
   
  * numba
   
  * pylobs
   
  * jupyter

## Informations

This course has been created for ["Journées SMAI-MODE 2022, Limoges"](https://indico.math.cnrs.fr/event/6564/)

## Affiliations and websites of the authors 

[Nelly Pustelnik](http://perso.ens-lyon.fr/nelly.pustelnik/): CNRS, Laboratoire de Physique, ENS de Lyon, France and INMA, UCLouvain, Belgium

[Audrey Repetti](https://sites.google.com/view/audreyrepetti) : Heriot-Watt University, Maxwell Institute, Edinburgh, UK

## Installation

```bash
conda create -n prox_tutorial --file Python_tutorial/requirement.txt
conda activate prox_tutorial
pip install bm3d
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# for drunet, retrieve pre-trained network from https://github.com/cszn/DPIR/tree/master/model_zoo (with wget)
mkdir Python_tutorial/checkpoint && cd checkpoint
# see https://stackoverflow.com/questions/37453841/download-a-file-from-google-drive-using-wget
wget "drive.google.com/u/3/uc?id=1oSsLjPPn6lqtzraFZLZGmwP_5KbPfTES&export=download&confirm=yes" -O drunet_gray.pth

# run the notebook from jupyterlab
jupyter lab
```

Note: Ubuntu users may have to install Open-BLAS to be able to use BM3D, using for instance the following command

```bash
sudo apt update
sudo apt install libopenblas-base
```
