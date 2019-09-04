# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:38:00 2019

@author: Diaa Elsayed
"""


import os 
import argparse

import numpy as np

import torch
from torch.autograd import Variable

import imageio

from models import Generator, Discriminator
from helpers import sample_image
from setup import train_loader


def manage():
    
    parser = argparse.ArgumentParser(description='AC-GAN is an is an \
                                     implementation of is an implementation \
                                     for Auxiliary Classifier with Generative \
                                     Adversarial Network')
    
    parser.add_argument('--cuda', type=bool, default=True, help='set this\
                        parameter to True value if you want to use cuda gpu,\
                        default is True')  
    
    parser.add_argument('--model_path', type=str, default='model', help='path \
                        for directory to save model on it')   
    
    parser.add_argument('--n_classes', type=int, default=10, help='number of \
                        classes of the dataset')
    
    parser.add_argument('--outputs_path', type=str, default='outputs', help='path\
                        for directory to add the generated images on it,\
                        if you don\'t use it output directory will created and \
                        add generated images on it')
    
    parser.add_argument('--img_shape', type=int, nargs='+',default=[1,28,28], help='\
                        list the shape of the images in order channels, height, weight')
   
    parser.add_argument('--embed_dim', type=int, default=128, help='number of embedding layer output')
    
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate value')

    parser.add_argument('--betas', type=int, nargs='+',default=[0.5, 0.999], 
                        help='values of beta1 and beta2 for adam optimizer')
    
    parser.add_argument('--n_epochs', type=int, default=5, help='number of \
                        epochs of training process')
    
    parser.add_argument('--sample_interval', type=int, default=1,
                        help='interval number ot save images while training')
    
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='number of epochs of early stopping')

    return parser.parse_args()

args = manage()

cuda = args.cuda and torch.cuda.is_available()
model_path = args.model_path
n_classes = args.n_classes
outputs_path = args.outputs_path
image_shape = args.img_shape
embed_dim = args.embed_dim
lr = args.lr
b1, b2 = args.betas
n_epochs = args.n_epochs
sample_interval = args.sample_interval
early_stopping = args.early_stopping

if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

if not os.path.exists(model_path):
    os.mkdir(model_path)



#Models
G = Generator(n_classes, embed_dim, image_shape)
D = Discriminator(n_classes, image_shape)

#losses Functions
adv_loss = torch.nn.BCELoss()
aux_loss = torch.nn.CrossEntropyLoss()


if cuda:
    print("Cuda is Available")
    G.cuda()
    D.cuda()

    adv_loss.cuda()
    aux_loss.cuda()


#Data types
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

#Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

cfg = {"n_classes":n_classes,
       "embed_dim":embed_dim,
       "image_shape":image_shape}

min_G_valid_loss = np.Inf
min_D_valid_loss = np.Inf

saved_epoch = 0

for epoch in range(n_epochs):
    G_valid_loss = 0
    D_valid_loss = 0
    for i, (imgs,labels) in enumerate(train_loader):
        
        batch_size = imgs.shape[0]
        
        real = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        gen = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)    
        
        real_labels = Variable(labels.type(LongTensor))
        real_imgs = Variable(imgs.type(FloatTensor))

        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
        noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, embed_dim))))


        optimizer_G.zero_grad()

        gen_imgs = G(gen_labels, noise)

        adv_score, aux_score = D(gen_imgs)

        G_loss = (adv_loss(adv_score, real) + aux_loss(aux_score, gen_labels)) / 2.
        
        G_loss.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()

        real_adv_score, real_aux_score = D(real_imgs)
        real_loss = (adv_loss(real_adv_score, real) + aux_loss(real_aux_score, real_labels)) / 2

        gen_adv_score, gen_aux_score = D(gen_imgs.detach())
        gen_loss = (adv_loss(gen_adv_score, gen) + aux_loss(gen_aux_score, gen_labels)) / 2

        D_loss = (real_loss + gen_loss) / 2.
        
        D_loss.backward()
        optimizer_D.step()

        pred = np.concatenate([real_aux_score.data.cpu().numpy(), gen_aux_score.data.cpu().numpy()], axis=0)
        gt = np.concatenate([real_labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        print(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, n_epochs, i, len(train_loader), D_loss.item(), 100 * d_acc, G_loss.item()), end=''
        )
       
        batches_done = epoch * len(train_loader) + i
        if batches_done % sample_interval == 0:
            sample_image(G, FloatTensor, LongTensor, embed_dim, n_classes, batches_done, outputs_path)
        
        G_valid_loss += G_loss.item()
        D_valid_loss += D_loss.item()
        
        
    G_valid_loss /= len(train_loader)
    D_valid_loss /= len(train_loader)
            
    if min_G_valid_loss > G_valid_loss:
        print ('\nG Validation loss decreased ({:.6f} --> {:.6f}). \
        Saving model ...\n'.format(min_G_valid_loss, G_valid_loss))
        min_valid_loss = G_valid_loss
        cfg['G_weights'] = G.state_dict()
        torch.save(cfg,os.path.join(model_path,'cfg.pth'))
        saved_epoch = epoch
    if min_D_valid_loss > D_valid_loss:
        print ('\nD Validation loss decreased ({:.6f} --> {:.6f}). \
        Saving model ...\n'.format(min_D_valid_loss, D_valid_loss))
        min_D_valid_loss = D_valid_loss
        cfg['D_weights'] = D.state_dict()
        torch.save(cfg,os.path.join(model_path,'cfg.pth'))
    if epoch - saved_epoch >= early_stopping:
        print("Early Stopping")
        break


with imageio.get_writer('AC_GAN.gif', mode='I') as writer:
  
  files = os.listdir(outputs_path)
  new_files = [int(f[:-4]) for f in files]
  filenames = [os.path.join(outputs_path, str(f)+'.png') for f in new_files]

  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
    
os.system('cp dcgan.gif AC_GAN.gif.png')