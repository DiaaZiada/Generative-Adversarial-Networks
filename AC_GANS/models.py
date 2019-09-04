# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:14:48 2019

@author: Diaa Elsayed
"""
import math

import torch
from torch import nn

from helpers import View


class Generator(nn.Module):
    def __init__(self,n_classes, embed_dim, image_shape):
        super(Generator, self).__init__()

        c, h, w = image_shape
        init_img_size = h // 4
        self.embedding = nn.Embedding(num_embeddings=n_classes, embedding_dim= embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim * init_img_size ** 2)
        self.relu = nn.ReLU()
        self.us = nn.Upsample(scale_factor=2)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.view = View((-1, embed_dim, init_img_size, init_img_size))

        self.bn1 = nn.BatchNorm2d(embed_dim, momentum=0.8)
        self.conv1 = nn.Conv2d(embed_dim, 128, 3, 1, 1)

        self.bn2 = nn.BatchNorm2d(128, momentum=0.8)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)

        self.bn3 = nn.BatchNorm2d(64, momentum=0.8)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)

        self.bn4 = nn.BatchNorm2d(32, momentum=0.8)
        self.conv4 = nn.Conv2d(32, c, 3, 1, 1)

    def forward(self, labels, noise):
        
        x = self.embedding(labels)
        x = torch.mul(x, noise)
        x = self.linear(x)
        x = self.relu(x)
        x = self.view(x)

        x = self.bn1(x)
        x = self.us(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.us(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        
        x = self.bn4(x)
        x = self.leakyrelu(x)
        x = self.conv4(x)

        x = self.tanh(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, bool_bn=True):
        super(Block, self).__init__()
        
        self.bool_bn = bool_bn
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(0.25)

        if self.bool_bn:
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        if self.bool_bn:
            x = self.bn(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_classes, image_shape):
        super(Discriminator, self).__init__()
        c, h, w = image_shape
        
        self.block1 = Block(c, 16, bool_bn=False)
        self.block2 = Block(16, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 128)

        downed_size = int(math.ceil(h / 2**4))

        self.view = View((-1, 128 * downed_size ** 2))

        self.advv_layer = nn.Linear(128 * downed_size ** 2 , 1)
        self.aux_layer = nn.Linear(128 * downed_size ** 2 , n_classes)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, img):
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)   

        x = self.view(x)

        adv = self.advv_layer(x)
        adv = self.sigmoid(adv)

        aux = self.aux_layer(x)
        aux = self.softmax(aux)
        
        return adv, aux
        