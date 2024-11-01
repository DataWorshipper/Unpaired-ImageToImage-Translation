import numpy as np 
import itertools
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch

from matplotlib import pyplot as plt
from IPython.display import clear_output
from PIL import Image
import matplotlib.image as mpimg
from utils import *
from cyclegan import *
cuda= "True" if torch.cuda.is_available() else False
print("USING CUDA" if cuda else "Not Using CUDA")

Tensor=torch.cuda.FloatTensor if cuda else torch.Tensor


class Hyperparameter(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

hp=Hyperparameter(
    epoch=0,
    n_epochs=200,
    dataset_train_mode='train',
    dataset_test_mode='test',
    batch_size=4,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    n_critic=5,
    sample_interval=100,
    num_res_blocks=19,
    lambda_cyc=10.0,
    lambda_id=5.0,
)

root_path="/content/drive/MyDrive/CycleGAN_Dataset"

def show_img(img,size=10):
    img=img/2+0.5
    npimg=img.numpy()
    plt.figure(figsize=(size,size))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def to_img(x):
    x=x.view(x.size(0)*2,hp.channels,hp.img_size,hp.img_size)
    return x

def plot_output(path,x,y):
    img=mpimg.imread(path)
    plt.figure(figsize=(x,y))
    plt.imshow(img)
    plt.show()

transforms=[transforms.Resize(hp.img_size,hp.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]

train_dataloader=DataLoader(
    ImageDataset(root=root_path,mode=hp.dataset_train_mode,transform=transforms),
    batch_size=hp.batch_size,
    shuffle=True,
    num_workers=hp.n_cpu)

val_dataloader==DataLoader( ImageDataset(root=root_path,mode=hp.dataset_test_mode,transform=transforms),
    batch_size=16,
    shuffle=True,
    num_workers=hp.n_cpu)


def save_img_samples(batches_done):
    """Saves a generated sample from the test set"""
    print("batches_done ", batches_done)
    imgs = next(iter(val_dataloader))

    Gen_AB.eval()
    Gen_BA.eval()

    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = Gen_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = Gen_BA(real_B)
 
    real_A = make_grid(real_A, nrow=16, normalize=True)
    real_B = make_grid(real_B, nrow=16, normalize=True)
    fake_A = make_grid(fake_A, nrow=16, normalize=True)
    fake_B = make_grid(fake_B, nrow=16, normalize=True)
  
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    path = root_path + "/%s.png" % (batches_done)  

   
    save_image(image_grid, path, normalize=False)
    return path


criterion_GAN=torch.nn.MSELoss()
criterion_cycle=torch.nn.L1Loss()
criterion_identity=torch.nn.L1Loss()

gen_AB=Generator(hp.input_shape,hp.num_res_blocks)
gen_BA=Generator(hp.input_shape,hp.num_res_blocks)
disc_A=Disciminator(hp.input_shape)
disc_B=Disciminator(hp.input_shape)

if cuda:
    gen_AB=gen_AB.cuda()
    gen_BA=gen_BA.cuda()
    disc_A=disc_A.cuda()
    disc_B=disc_B.cuda()
    criterion_GAN=criterion_GAN.cuda()
    criterion_cycle=criterion_cycle.cuda()
    criterion_identity=criterion_identity.cuda()

gen_AB.apply(initialize_conv_weights_normal)

gen_BA.apply(initialize_conv_weights_normal)

disc_A.apply(initialize_conv_weights_normal)    

disc_B.apply(initialize_conv_weights_normal)

fake_A_buffer=ReplayBuffer()
fake_B_buffer=ReplayBuffer()

optimizer_G=torch.optim.Adam(itertools.chain(gen_AB.parameters(),gen_BA.parameters()),lr=hp.lr,betas=(hp.b1,hp.b2))
optimizer_disc_A=torch.optim.Adam(disc_A.parameters(),lr=hp.lr,betas=(hp.b1,hp.b2))
optimizer_disc_B=torch.optim.Adam(disc_B.parameters(),lr=hp.lr,betas=(hp.b1,hp.b2))

lr_scheduler_G=torch.optim.lr_scheduler.LambdaLR(
    optimizer_G,
    lr_lambda=LambdaLR(hp.n_epochs,hp.epoch,hp.decay_start_epoch).step)

lr_scheduler_disc_A=torch.optim.lr_scheduler.LambdaLR(
    optimizer_disc_A,
    lr_lambda=LambdaLR(hp.n_epochs,hp.epoch,hp.decay_start_epoch).step)
lr_scheduler_disc_B=torch.optim.lr_scheduler.LambdaLR(
    optimizer_disc_B,
    lr_lambda=LambdaLR(hp.n_epochs,hp.epoch,hp.decay_start_epoch).step)

def train(gen_AB,gen_BA,disc_A,disc_B,train_dataloader,n_epochs,criterion_identity,criterion_cycle,criterion_GAN,lambda_cyc,optimizer_G,fake_A_buffer,fake_B_buffer,clear_output,optimizer_disc_A,optimizer_disc_B,Tensor,sample_interval,lambda_id):
    prev_time=time.time()
    for epoch in range(hp.epoch,hp.n_epochs):
        for i,batch in enumerate(train_dataloader):
            real_A=Variable(batch["A"].type(Tensor))
            real_B=Variable(batch["B"].type(Tensor))
            valid=Variable(np.ones(real_A.shape[0],*disc_A.output_shape),requires_grad=False)
            fake=Variable(np.zeros(real_A.shape[0],*disc_A.output_shape),requires_grad=False)
            gen_AB.train()
            gen_BA.train()
            optimizer_G.zero_grad()
            loss_A_id=criterion_identity(gen_BA(real_A),real_A)
            loss_B_id=criterion_identity(gen_AB(real_B),real_B)
            loss_id=(loss_A_id+loss_B_id)/2
            fake_B=gen_AB(real_A)
            loss_A_g=criterion_GAN(disc_B(fake_B),valid)
            fake_A=gen_BA(real_B)
            loss_B_g=criterion_GAN(disc_A(fake_A),valid)
            loss_GAN=(loss_A_g+loss_B_g)/2
            loss_cycle_A=criterion_cycle(gen_BA(fake_B),real_A)
            loss_cycle_B=criterion_cycle(gen_AB(fake_A),real_B)
            loss_cycle=(loss_cycle_A+loss_cycle_B)/2
            loss_G=lambda_id*loss_id+loss_GAN+lambda_cyc*loss_cycle
            loss_G.backward()
            optimizer_G.step()

            optimizer_disc_A.zero_grad()
            fake_A=fake_A_buffer.push_and_pop(fake_A)
            loss_D_A=(criterion_GAN(disc_A(real_A),valid)+criterion_GAN(disc_A(fake_A.detach()),fake))/2
            loss_D_A.backward()
            optimizer_disc_A.step()

            optimizer_disc_B.zero_grad()
            fake_B=fake_B_buffer.push_and_pop(fake_B)
            loss_D_B=(criterion_GAN(disc_B(real_B),valid)+criterion_GAN(disc_B(fake_B.detach()),fake))/2
            loss_D_B.backward()
            optimizer_disc_B.step()

            loss_D=(loss_D_A+loss_D_B)/2

            batches_done = epoch * len(train_dataloader) + i

            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_id.item(),
                    time_left,
                )
            )

            
            if batches_done % sample_interval == 0:
                clear_output()
                plot_output(save_img_samples(batches_done), 30, 40)

            



train(
    gen_AB,
    gen_BA,
    disc_A,
    disc_B,
    train_dataloader,
    hp.n_epochs,
    criterion_identity,
    criterion_cycle,
    criterion_GAN,
    hp.lambda_cyc,
    optimizer_G,
    fake_A_buffer,
    fake_B_buffer,
    clear_output,
    optimizer_disc_A,
    optimizer_disc_B,
    Tensor,
    sample_interval,
    hp.lambda_id
)

        





