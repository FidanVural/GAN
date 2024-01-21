# GAN (Generative Adversarial Networks)
In this repo, you will be finding codes about different kind of GANs and some results that obtained from the experiments. 
First of all, let's start with the installation process.

## Installation 

```bash
   git clone https://github.com/FidanVural/GAN.git
   cd GAN/ 
   sudo pip3 install -r requirements.txt
```
 
Before we start, if you want to download the datasets, you have to change "download" parameter to True in the codes. Let's look at this line: **dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)**. After the first download, you can change the "dowload" to False. 

## GAN (Generative Adversarial Networks)  

GANs have two structures: the Generator and the Discriminator. While generator tries to create a new image (a fake image) that looks like the real one from random noise, discriminator tries to distinguish between a real image and a fake image. These two structures always try to fool each other.
<p align="center">
  <img width="500" height="200" src="https://github.com/FidanVural/GAN/assets/56233156/6d009b53-a327-495f-8bb8-ada95a370b0d">
</p> 

Also, you can see the objective functions of the generator and the discriminator in the below images. The generator tries to minimize the loss function, whereas the discriminator tries to maximize it. In the implementation you can see that BCE loss was used. The generator wants to minimize the function because it wants to fool the discriminator. That's why, it wants D(G(z)) value to be 1 or close to 1 for reaching the aim that minimizes the function. On the other hand, the discriminator wants to maximize the function because it wants to detect the created image as fake. Therefore, it wants D(G(z)) value to be 0 or close to 0. You can find more graphics and explanation on [my whiteboard pdf](https://drive.google.com/drive/folders/1EIEWs1vZnrzZOywFz4OM91QTLSuoySqh?usp=sharing).

<p align="center">   
  <img width="500" height="80" src="https://github.com/FidanVural/GAN/assets/56233156/93a1faaf-dff4-444f-af2d-57c7cc1f8fcf">
</p>  
<p align="center"> 
    <em>Generator objective function</em> 
</p>  

<p align="center"> 
  <img width="700" height="80" src="https://github.com/FidanVural/GAN/assets/56233156/bfbbdaa8-9326-46ad-b471-9b143c27c0be">
</p>
<p align="center"> 
    <em>Discriminator objective function</em>
</p>

After training the GAN 50 epochs on MNIST dataset, results are shown in the below. 

Target Images             |  Fake Images
:------------------------:|:-------------------------:
![gan_true](https://github.com/FidanVural/GAN/assets/56233156/15d43c02-95ea-4c68-98c7-ca212cf3e7b8) | ![gan_fake](https://github.com/FidanVural/GAN/assets/56233156/c792a816-10f2-4384-88a7-a72007c6ca47)   

By tweaking the hyperparameters a bit, we can get better results.


Target Images             |  Fake Images
:------------------------:|:-------------------------:
![gan_real_opt](https://github.com/FidanVural/GAN/assets/56233156/ebb9816a-e815-4ba8-a419-7263d69e5fbd) | ![gan_fake_opt](https://github.com/FidanVural/GAN/assets/56233156/148ca590-4795-494b-93ec-a925deee9989)


## DCGAN (Deep Convolutional Generative Adversarial Networks)
DCGAN is so similar to the GAN. Main difference between the GAN and the DCGAN is the use of convolutional blocks. You can explore [dcgan.py](https://github.com/FidanVural/GAN/blob/master/dcgan.py) and [dcgan_training.py](https://github.com/FidanVural/GAN/blob/master/dcgan_training.py) scripts. Due to the utilization of convolutional layers, training process is slower than the GAN but results surpass it. You can take a look at the results below (after 20 epochs of training). 

Target Images             |  Fake Images
:------------------------:|:-------------------------:
![dcgan_real_opt](https://github.com/FidanVural/GAN/assets/56233156/67b76a97-fc6e-497d-8b57-4fa245dd30e1) | ![dcgan_fake_opt](https://github.com/FidanVural/GAN/assets/56233156/b3fcd651-cc1c-4c3c-b3cb-574d4f3def5a)


## WGAN (Wasserstein GAN)

Binary Cross Entropy (BCE) loss can cause some problems when we train our models. One of the problems is vanishing gradient. Our discriminator model is generally the fast learner while the generator is more difficult to train. That's why, the discriminator can distiguish the real and fake images easily and it doesn't give efficient feedbacks to the generator. The generator cannot improve itself. Because of that W-Loss was proposed. W-Loss is nothing but the difference between the expected values of the predictions of the discriminator which is called here "critic". While discriminator wants to maximize this loss, generator wants to minimize it. Moreover, the last layer of discriminator is linear layer instead of sigmoid. That's why, the discriminator can give any number as output.  You can take a look to the W-Loss and difference between the discriminator outputs below. 


<p align="center"> 
  <img width="550" height="120" src="https://github.com/FidanVural/GAN/assets/56233156/8e7a8cde-9032-4766-87fc-cd38d28bd7f9">
</p>
<p align="center"> 
  <img width="580" height="200" src="https://github.com/FidanVural/GAN/assets/56233156/b5716dd4-f94c-4f48-9433-171fabe0f927">
</p>  

We require a limit for predictions of the discriminator because the discriminator outputs can be get any value. Therefore, to apply WGAN, we have 2 different approach. One of them is WGAN with weight clipping and the other one is WGAN with gradient penalty. In the WGAN paper, author says like weight clipping is a terrible way to do this. But, you can find WGAN with weight clipping codes in [wgan.py](https://github.com/FidanVural/GAN/blob/master/wgan.py) and [wgan_training.py](https://github.com/FidanVural/GAN/blob/master/wgan_training.py). Also, you can find WGAN with gradient penalty codes in [wgan_gp.py](https://github.com/FidanVural/GAN/blob/master/wgan_gp.py), [wgan_gp_training.py](https://github.com/FidanVural/GAN/blob/master/wgan_gp_training.py) and [utils_gp.py](https://github.com/FidanVural/GAN/blob/master/utils_gp.py). You can take a look [my whiteboard pdf](https://drive.google.com/drive/folders/1EIEWs1vZnrzZOywFz4OM91QTLSuoySqh?usp=sharing) for more detail.

You can see the training results below after 15 epochs.
Target Images             |  Fake Images
:------------------------:|:-------------------------:
![wgan_r](https://github.com/FidanVural/GAN/assets/56233156/c0404e64-6587-41e9-acbf-de61dd1f5bc5) | ![wgan_f](https://github.com/FidanVural/GAN/assets/56233156/4614fe40-61da-4265-aabd-c4a8d3b4286e)


## CGAN (Conditional GAN)
So far, we looked at the unconditional GANs. There is a metaphor to explain the difference between conditional and unconditinal GAN. We can regard unconditional GANs as a gumball machine because you get outputs from a random class. On the other hand, we can deem conditional GAN as a vending machine because you get whatever you want. That's why, training dataset must have been labeled in the CGAN. We add class information both the generator and the critic. It is added to extra channel for the discriminator and it is added to the noise vector for the generator. If you change the generator noise vector you can obtain different image for same category. But if you change the class you can get different image from the chosen category.
Below images show the outputs of the training.

You can see the training results below.
Target Images             |  Fake Images
:------------------------:|:-------------------------:
![cgan_r](https://github.com/FidanVural/GAN/assets/56233156/2a33d5a1-a453-4db5-b4b9-1c91a51df717) | ![cgan_f](https://github.com/FidanVural/GAN/assets/56233156/921cde47-c998-4552-a1d4-a78d95306627)

I'll update this page as I learn new things.

## PIX2PIX (Image-to-Image Translation with GAN)
Pix2pix is a general solution for image to image translation. Generally, if you want to do image to image translation, you use different models and loss functions. To overcome this problem, pix2pix presents a general-purpose solution for many image to image tasks like image colorizing, reconstructing objects from edge maps and synthesizing photos from label maps. It also has a generator and discriminator. U-Net based architecture is used for the generator and PatchGAN is used for the discriminator. Pix2pix is a conditinal GANs and because of that the discriminator also takes y as an input. This training procedure can be seen below.

<p align="center">   
  <img src="https://github.com/FidanVural/GAN/assets/56233156/6ca0ba4e-2762-48e8-ac96-15f656302bd2">
</p>  
<p align="center"> 
    <em>Training Procedure [7]</em> 
</p>

In the paper, authors also say that L1 loss is preferred instead of L2 because L1 loss encourges less blurring. When it comes to the losses, they can decide two different losses -> adversarial loss and L1 loss. You can look at the losses in the below. If you look at the adversarial loss, you can see that noise (z) is given to the generator. However, noise was provided in the form of dropout instead of giving it as an input and you can see this in the code like that.  

<p align="center">   
  <img src="https://github.com/FidanVural/GAN/assets/56233156/223193ec-fe96-4246-b259-cda7cbe7d932">
</p>  
<p align="center"> 
    <em>Adversarial Loss [7]</em> 
</p>  

<p align="center">   
  <img src="https://github.com/FidanVural/GAN/assets/56233156/64455ea7-1f17-4f7e-a84f-fce6566c9aa9">
</p>  
<p align="center"> 
    <em>Overall Loss [7]</em> 
</p>  

<p align="center">   
  <img src="https://github.com/FidanVural/GAN/assets/56233156/8d59b32f-2490-4360-beaa-0da796d71b29">
</p>  
<p align="center"> 
    <em>Input Images</em> 
</p>  

<p align="center">   
  <img src="https://github.com/FidanVural/GAN/assets/56233156/68d0b5b4-2480-4efb-b30b-79d0dfb74843">
</p>  
<p align="center"> 
    <em>Real Images</em> 
</p>  

<p align="center">   
  <img src="https://github.com/FidanVural/GAN/assets/56233156/fd1b9a9f-d233-4589-92ec-76df73471b4e">
</p>  
<p align="center"> 
    <em>Colorized Images</em> 
</p>  


### RESOURCES

[1] https://arxiv.org/pdf/1406.2661.pdf

[2] https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans

[3] https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09

[4] https://towardsdatascience.com/generative-adversarial-networks-explained-34472718707a

[5] https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/ch04.html

[6] https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490

[7] https://arxiv.org/pdf/1611.07004.pdf
