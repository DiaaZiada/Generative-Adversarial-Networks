# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:18:11 2019

@author: Diaa Elsayed
"""
import os 
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class View:
    def __init__(self, reshape):
        self.reshape = reshape

    def __call__(self, x):
        return x.view(*self.reshape)
    
def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)).squeeze())

def sample_image(G, FloatTensor, LongTensor, embed_dim, n_row, batches_done, outputs_path):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    noise = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, embed_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = G(labels, noise)
    save_image(gen_imgs.data,os.path.join(outputs_path,"%d.png" % batches_done), nrow=n_row, normalize=True)
