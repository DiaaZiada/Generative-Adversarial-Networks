# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:21:40 2019

@author: Diaa Elsayed
"""
import os 
import argparse

import torch
from torchvision.utils import save_image

import matplotlib.pyplot as plt

from models import Generator
from helpers import imshow



def manage():
    
    parser = argparse.ArgumentParser(description='AC-GAN is an is an \
                                     implementation of Auxiliary\
                                     Classifier with Generative \
                                     Adversarial Network')
    parser.add_argument('--cuda', type=bool, default=True, help='set this\
                        parameter to True value if you want to use cuda gpu,\
                        default is True')   
    parser.add_argument('--model_path', type=str, default='model', help='path \
                        for directory contains pytorch model')   
    parser.add_argument('--classes', type=int, nargs='+',default=[-1], help='\
                        list the classes number you want to generat ,\n if you \
                        to generat one sample from every class pass -1')
    parser.add_argument('--outputs_path', type=str, default='outputs', help='path\
                        for directory to add the generated images on it,\
                        if you don\'t use it output directory will created and \
                        add generated images on it')
    return parser.parse_args()


args = manage()

cuda = args.cuda and torch.cuda.is_available()

outputs_path = args.outputs_path

cfg_path = os.path.join(args.model_path, 'cfg.pth')

cfg = torch.load(cfg_path)

n_classes = cfg["n_classes"]
embed_dim = cfg["embed_dim"]
image_shape = cfg["image_shape"]
G_weights = cfg["G_weights"]

G = Generator(n_classes, embed_dim, image_shape)

classes = args.classes

if len(classes) == 1 and classes[0] == -1:    
    classes = list(range(n_classes))

labels = torch.LongTensor([classes])
noise = torch.rand(1,1,embed_dim)

if cuda:    
    G.load_state_dict(G_weights)
    G.cuda()
    labels = labels.cuda()
    noise = noise.cuda()
else:
    G.load_state_dict(G_weights)

images= G(labels[None],noise)

for i in range(len(images)):
    imshow(images[i].detach().numpy())
    plt.show()

if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)
else:
    image_number = len(os.listdir(outputs_path))
    
save_image(images.data, os.path.join(outputs_path,"%d.png"% image_number), nrow=1, normalize=True)

