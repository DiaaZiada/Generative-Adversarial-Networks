# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:43:18 2019

@author: Diaa Elsayed
"""
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
                               torchvision.transforms.Normalize(
                                 (0.5,), (0.5,))
                             ])),
    
  batch_size=batch_size_train, shuffle=True)



