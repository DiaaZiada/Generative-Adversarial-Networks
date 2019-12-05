
# Adversarial Autoencoders
this subrepo is an implementation for Adversarial Autoencoders
**AAE-GAN Architecture**

![AAE-GAN Architecture](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/aaegan.png)

# Generative Adversarial Network

**Model Improvment Process**

![AAE_GAN_images](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/AAE_GAN.gif)

![AAE_GAN_graph](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/AAE_GAN_graph.gif)

from

![zero](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/0%20%281%29.png)

to

![final](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/138%20%281%29.png)

and from

![zero](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/0.png)

to

![final](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/138.png)

****Requirements****
 - [Python](https://www.python.org/) 3.*
 - [OpenCV](https://opencv.org)
 - [Pytorch](https://pytorch.org/)
 
**SubRepo Contains:**

 - [Notebook](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/aae.ipynb):  contains steps of train and test processes using [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
 - Run script: to test the model by generating some images


## Run Script
### Args Options
```
usage: run.py [-h] [--cuda CUDA] [--encoder_path ENCODER_PATH]
              [--decoder_path DECODER_PATH] [--image_path IMAGE_PATH]
              [--xy_values XY_VALUES [XY_VALUES ...]]

AAE-GAN is an is an implementation of Adversarial Autoencoders

optional arguments:
  -h, --help            show this help message and exit
  --cuda CUDA           set this parameter to True value if you want to use
                        cuda gpu, default is True
  --encoder_path ENCODER_PATH
                        path weights of the encoder
  --decoder_path DECODER_PATH
                        path weights of the decoder
  --image_path IMAGE_PATH
                        path for the image that you want to in code and decode
  --xy_values XY_VALUES [XY_VALUES ...]
                        x, y values from the disruption to decodding
```
**Examples of Test Script generated images**

original![original](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/orginal.png)

after encoding and decoding
![aeimage](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/encodeddecoded.png)

generated using distribution values (0, -10)
![enter image description here](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/dist.png)

## Credits
[Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) paper
