#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:00:31 2019

@author: diaa
"""

import argparse

import torch
from torchvision import transforms

import cv2

from models import Encoder, Decoder



def manage():
    
    parser = argparse.ArgumentParser(description='AAE-GAN is an is an \
                                     implementation of Adversarial Autoencoders')
    parser.add_argument('--cuda', type=bool, default=True, help='set this\
                        parameter to True value if you want to use cuda gpu,\
                        default is True')   
    parser.add_argument('--encoder_path', type=str, default='models/encoder.pth', help='path \
                        weights of the encoder')    
    parser.add_argument('--decoder_path', type=str, default='models/decoder.pth', help='path \
                        weights of the decoder')   
    parser.add_argument('--image_path', type=str, help='path\
                        for the image that you want to in code and decode')
    parser.add_argument('--xy_values', type=float, nargs='+', help=' x, y\
                        values from the disruption to decodding')
    return parser.parse_args()

args = manage()

cuda = args.cuda and torch.cuda.is_available()    

encoder = Encoder()
decoder = Decoder()

if cuda:    
    encoder.load_state_dict(torch.load(args.encoder_path))
    encoder.cuda()
else:
    encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))


if cuda:    
    decoder.load_state_dict(torch.load(args.decoder_path))
    decoder.cuda()
else:
    decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu'))

if args.image_path:
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    code = encoder(transforms.ToTensor()(transforms.ToPILImage()(image)).view(-1,784))
    print('code: ',code )
    decode = decoder(code)
    image2 = decode.cpu().view(28,28).detach().numpy()
    cv2.imshow('orginal image',image)
    cv2.imshow('image after encoding and decoding',image2)
    cv2.waitKey(0)
if args.xy_values:
    decode = decoder(torch.FloatTensor(args.xy_values)[None])
    image = decode.cpu().view(28,28).detach().numpy()
    cv2.imshow('image after decoding',image)
    cv2.waitKey(0)    


