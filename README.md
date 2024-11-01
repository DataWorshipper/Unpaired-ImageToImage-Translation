# CycleGAN -Unpaired ImageToImage Translation

This repository provides an implementation of CycleGAN for unpaired image-to-image translation, specifically using the Yosemite summer-to-winter dataset. This project demonstrates how to convert images taken in the summer season to winter scenes, and vice versa, using unpaired datasets from both domains.

# Project Overview
CycleGAN enables unpaired image-to-image translation by training on two separate datasets (one for each domain) without needing aligned image pairs. Cycle consistency, adversarial, and identity losses work together to ensure that images retain structure while taking on the visual properties of the target domain.

# How It Works
CycleGAN consists of:

Two Generators:

G: Transforms images from domain X to domain Y.

F: Transforms images from domain Y to domain X.

Two Discriminators:

D_X: Distinguishes real images in domain X from generated images.

D_Y: Distinguishes real images in domain Y from generated images.

# Key Losses

Cycle Consistency Loss: Ensures that an image translated to another domain and then back to the original domain remains unchanged.

Adversarial Loss: Encourages each generator to produce images indistinguishable from real images in the target domain.

Identity Loss: Helps retain color and content similarity when an image from one domain closely resembles the target domain.

# Input Data


The input data for this CycleGAN model includes Yosemite National Park images captured in two distinct seasons: summer and winter. This dataset is split into two domains:

Domain A (Summer): Images of Yosemite landscapes in summer.

Domain B (Winter): Images of Yosemite landscapes in winter.

The CycleGAN model is trained to learn how to translate between these two domains, effectively generating winter versions of summer images and vice versa. This dataset can be downloaded from Kaggle.  

# Output 


![CYCLE_GAN_IMGES](https://github.com/user-attachments/assets/cb2ea1e8-8c49-4dad-8cb7-c0990e7be642)

This is how the output looks like after training the model for roughly 200 epochs


