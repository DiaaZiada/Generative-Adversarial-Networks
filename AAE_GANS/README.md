
# Adversarial Autoencoders
this subrepo is an implementation for Adversarial Autoencoders
**AAE-GAN Architecture**

![AAE-GAN Architecture](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/aaegan.png)

# Generative Adversarial Network

**Model Improvment Process**

![AAE_GAN_images](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AAE_GANS/images/AAE_GAN%20(1).gif)

![AAE_GAN_graph](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/AC_GAN.gif)

****Requirements****
 - [Python](https://www.python.org/) 3.*
 - [OpenCV](https://opencv.org)
 - [Pytorch](https://pytorch.org/)
 
**SubRepo Contains:**

 - [Notebook](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/AC_GAN.ipynb): contains steps of train and test processes using [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
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

![generated image](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/1.png)
![generated image](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/2.png)
![generated image](https://github.com/DiaaZiada/Generative-Adversarial-Networks/blob/master/AC_GANS/images/3.png)

## Credits
[Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585) paper
