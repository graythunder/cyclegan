import os, sys
import datetime
import time
import numpy as np
import argparse

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image,make_grid

from dataset import MyDataset
from model import *

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='horse2zebra',help='Enter the dataset you want the model to train on')
parser.add_argument('--image_size',default=256,type=int,help='height and width of images')
parser.add_argument('--batch_size', default=1,type=int,help='Enter the batch size')
parser.add_argument('--n_epochs',default=200,type=int,help='Enter the total number of epochs')
parser.add_argument('--learning_rate',default=0.0002,type=float)
parser.add_argument('--cyc_lambda',default=10.0,type=float,help='weight of cycle loss')
args = parser.parse_args()

#networks
G = Generator(args.image_size) #A -> B
F = Generator(args.image_size) #B -> A 
D_A = Discriminator(args.image_size)
D_B = Discriminator(args.image_size)

#criterions
criterion_GAN = nn.MSELoss() 
criterion_cycle = nn.L1Loss()
#TODO: identity mapping loss

if cuda:
    G.cuda()
    F.cuda()
    D_A.cuda()
    D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()

#optimizer
optimizer_G = torch.optim.Adam(list(G.parameters())+list(F.parameters()), lr=args.learning_rate)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.learning_rate)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.learning_rate)


#dataloader
train_dataloader = DataLoader(MyDataset(dataname=args.dataset), batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(MyDataset(dataname=args.dataset, phase='test'))

print("Starting Training Loop...")
for epoch in range(args.n_epochs):
    for i, batch in enumerate(train_dataloader):
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
        real_label = Variable(torch.ones(args.batch_size,*D_A.out_shape), requires_grad=False).to(device)
        fake_label = Variable(torch.zeros(args.batch_size,*D_A.out_shape), requires_grad=False).to(device)
        
        #train generators
        optimizer_G.zero_grad()
        fake_A = F(real_B)
        fake_B = G(real_A)       
        loss_GAN = criterion_GAN(D_A(fake_A), real_label) + criterion_GAN(D_B(fake_B), real_label)
        loss_cycle = criterion_cycle(real_A, F(fake_B)) + criterion_cycle(real_B, G(fake_A))
        loss_gen = loss_GAN + loss_cycle * args.cyc_lambda
        loss_gen.backward()
        optimizer_G.step()

        #train discriminators
        #TODO: image buffer
        optimizer_D_A.zero_grad()
        loss_D_A = criterion_GAN(D_A(real_A), real_label) + criterion_GAN(D_A(fake_A.detach()), fake_label)
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_D_B = criterion_GAN(D_B(real_B), real_label) + criterion_GAN(D_B(fake_B.detach()), fake_label)
        loss_D_B.backward()
        optimizer_D_B.step() 

        #training log
        if i % 100 == 99:
            D_x = D_B(fake_B).mean().item()
            print('[%d/%d][%d/%d]\tLoss_GAN: %.4f\tLoss_cyc: %.4f\tLoss_D: %.4f\tD(G(x)): %.4f\t'
                        % (epoch+1, args.n_epochs, i+1, len(train_dataloader),
                            loss_gen.item(), loss_cycle.item(), (loss_D_A.item()+loss_D_B.item())/2, D_x))