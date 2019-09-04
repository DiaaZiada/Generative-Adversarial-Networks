
Welcome fileWelcome file
![GANs# Auxiliary Classifier with Generative Adversarial Network
this subrepo is an implementation for Auxiliary Classifier with Generative Adversarial Network

**AC-GAN Architecture**

![AC-GAN Architecture](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/acgans.jpg)

# Generative Adversarial Network
Collec.jpg)

**Model Improvment Process**

![AC_GAN](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/AC_GAN.gif)

****Requirements****
 - [Python](https://www.python.org/) 3.*
 - [Numpy](http://www.numpy.org/)
 - [Matplotlib](https://matplotlib.org)
 - [Imageio](https://pypi.org/project/imageio/)
 - [Pytorch](https://pytorch.org/)
 
**SubRepo Contains:**

 - [Notebook](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/AC_GAN.ipynb): contains steps of train and test processes using [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
 - Training script: to make train on your own dataset
 - Test script: to test the model by generating some images

## Train Script
### Setup
To Make train user custom dataset you have to define the dataloader for this data in [setup.py](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/setup.py) file by creating object `train_loader`
for example the repo by default will load [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
```py
import torch

import torchvision

batch_size_train = 64

random_seed = 1

torch.backends.cudnn.enabled = False

torch.manual_seed(random_seed)
train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('/files/', train=True, download=True,
	transform=torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,), (0.5,))])),
	batch_size=batch_size_train, shuffle=True)
```
### Args Options
```
usage: train.py [-h] [--cuda CUDA] [--model_path MODEL_PATH]
                [--n_classes N_CLASSES] [--outputs_path OUTPUTS_PATH]
                [--img_shape IMG_SHAPE [IMG_SHAPE ...]]
                [--embed_dim EMBED_DIM] [--lr LR] [--betas BETAS [BETAS ...]]
                [--n_epochs N_EPOCHS] [--sample_interval SAMPLE_INTERVAL]
                [--early_stopping EARLY_STOPPING]

AC-GAN is an is an implementation of is an implementations of for Auxiliary
Classifier with Generative Adversarial Networks (GANs) suggested in research papers using Pytorch framework
**Implementations:**

 - [Auxilia

optional arguments:
  -h, --help            show this help message and exit
  --cuda CUDA           set this parameter to True value if you want to use
                        cuda gpu, default is True
  --model_path MODEL_PATH
                        path for directory to save model on it
  --n_classes N_CLASSES
                        number of classes of the dataset
  --outputs_path OUTPUTS_PATH
                        path for directory to add the generated images on it,
                        if you don't use it output directory will created and
                        add generated images on it
  --img_shape IMG_SHAPE [IMG_SHAPE ...]
                        list the shape of the images in order channels,
                        height, weight
  --embed_dim EMBED_DIM
                        number of embedding layer output
  --lr LR               learning rate value
  --betas BETAS [BETAS ...]
                        values of beta1 and beta2 for adam optimizer
  --n_epochs N_EPOCHS   number of epochs of training process
  --sample_interval SAMPLE_INTERVAL
                        interval number ot save images while training
  --early_stopping EARLY_STOPPING
                        number of epochs of early stopping
```

## Test Script
### Args Options
```
AC-GAN is an is an implementation of is an implementation for Auxiliary
Classifier with Generative Adversarial Network

optional arguments:
  -h, --help            show this help message and exit
  --cuda CUDA           set this parameter to True value if you want to use
                        cuda gpu, default is True
  --model_path MODEL_PATH
                        path for directory contains pytorch model
  --classes CLASSES [CLASSES ...]
                        list the classes number you want to generat , if you
                        to generat one sample from every Cclassifier with  pass -1
  --outputs_path OUTPUTS_PATH
                        path for directory to add the generated images on it,
                        if you don't use it output directory will created and
                        add generated images on it
```
**Examples of Test Script generated images**

![generated image](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/1.png)
![generated image](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/2.png)
![generated image](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/3.png)

## Credits
[Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585) paper
[]()
![GANs# Auxiliary Classifier with Generative Adversarial Network
this subrepo is an implementation for Auxiliary Classifier with Generative Adversarial Network

AC-GAN Architecture

AC-GAN Architecture

Generative Adversarial Network
Collec.jpg)

Model Improvment Process

AC_GAN

Requirements

Python 3.*
Numpy
Matplotlib
Imageio
Pytorch
SubRepo Contains:

Notebook: contains steps of train and test processes using MNIST dataset
Training script: to make train on your own dataset
Test script: to test the model by generating some images
Train Script
Setup
To Make train user custom dataset you have to define the dataloader for this data in setup.py file by creating object train_loader
for example the repo by default will load MNIST dataset

import torch

import torchvision

batch_size_train = 64

random_seed = 1

torch.backends.cudnn.enabled = False

torch.manual_seed(random_seed)
train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('/files/', train=True, download=True,
	transform=torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,), (0.5,))])),
	batch_size=batch_size_train, shuffle=True)
Args Options
usage: train.py [-h] [--cuda CUDA] [--model_path MODEL_PATH]
                [--n_classes N_CLASSES] [--outputs_path OUTPUTS_PATH]
                [--img_shape IMG_SHAPE [IMG_SHAPE ...]]
                [--embed_dim EMBED_DIM] [--lr LR] [--betas BETAS [BETAS ...]]
                [--n_epochs N_EPOCHS] [--sample_interval SAMPLE_INTERVAL]
                [--early_stopping EARLY_STOPPING]

AC-GAN is an is an implementation of is an implementations of for Auxiliary
Classifier with Generative Adversarial Networks (GANs) suggested in research papers using Pytorch framework
**Implementations:**

 - [Auxilia

optional arguments:
  -h, --help            show this help message and exit
  --cuda CUDA           set this parameter to True value if you want to use
                        cuda gpu, default is True
  --model_path MODEL_PATH
                        path for directory to save model on it
  --n_classes N_CLASSES
                        number of classes of the dataset
  --outputs_path OUTPUTS_PATH
                        path for directory to add the generated images on it,
                        if you don't use it output directory will created and
                        add generated images on it
  --img_shape IMG_SHAPE [IMG_SHAPE ...]
                        list the shape of the images in order channels,
                        height, weight
  --embed_dim EMBED_DIM
                        number of embedding layer output
  --lr LR               learning rate value
  --betas BETAS [BETAS ...]
                        values of beta1 and beta2 for adam optimizer
  --n_epochs N_EPOCHS   number of epochs of training process
  --sample_interval SAMPLE_INTERVAL
                        interval number ot save images while training
  --early_stopping EARLY_STOPPING
                        number of epochs of early stopping
Test Script
Args Options
AC-GAN is an is an implementation of is an implementation for Auxiliary
Classifier with Generative Adversarial Network

optional arguments:
  -h, --help            show this help message and exit
  --cuda CUDA           set this parameter to True value if you want to use
                        cuda gpu, default is True
  --model_path MODEL_PATH
                        path for directory contains pytorch model
  --classes CLASSES [CLASSES ...]
                        list the classes number you want to generat , if you
                        to generat one sample from every Cclassifier with  pass -1
  --outputs_path OUTPUTS_PATH
                        path for directory to add the generated images on it,
                        if you don't use it output directory will created and
                        add generated images on it
Examples of Test Script generated images

![generated image](https://github.com/DiaaZiada/Generative-Adversarial -Networks/blob/master/AC_GANS/images/1.png)
generated image

blob/master/AC_GANS/images/2.png)
generated image

Credits
Conditional Image Synthesis With Auxiliary Classifier GANs paper

Markdown 5146 bytes 483 words 123 lines Ln 10, Col 0 HTML 3041 characters 454 words 90 paragraphs
FILE HISTORY
The following revisions are stored in Google Drive app data.
