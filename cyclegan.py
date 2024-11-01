import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

from torch.utils.data import Dataset
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_shape, num_res_blocks):
        super(Generator, self).__init__()
        channels = input_shape[0]
        out_channels = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, 7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        in_channels = out_channels
        
        for i in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            
        for i in range(num_res_blocks):
            model += [ResidualBlock(in_channels)]
        
        for i in range(2):
            out_channels //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_channels, channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model) 

    def forward(self, x):
        return self.model(x)

class Disciminator(nn.Module):
    def __init__(self,input_shape):
        super(Discriminator,self).__init__()
        channels,height,width=input_shape
        self.output_shape=(1,height //2**4,width //2**4)
        def discriminator_blocks(in_channels,out_channels,normalize=True):
            layers=[
                    nn.Conv2d(in_channels,out_channels,4,stride=2,padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.model=nn.Sequential(
            discriminator_blocks(channels,out_channels=64,normalize=False),
            discriminator_blocks(64,out_channels=128),
            discriminator_blocks(128,out_channels=256),
            discriminator_blocks(256,out_channels=512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512,out_channels=1,4,padding=1))
    def forward(self,img):
        return self.model(img)


        



            

