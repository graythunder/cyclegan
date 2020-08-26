import os, sys
import datetime
import time
import numpy as np
import argparse
from itertools import chain

import torch
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
parser.add_argument('--n_workers', default=1,type=int,help='number of cpu threads ')
parser.add_argument('--n_epochs',default=200,type=int,help='Enter the total number of epochs')
parser.add_argument('--learning_rate',default=0.0002,type=float)
parser.add_argument('--cyc_lambda',default=10.0,type=float,help='weight of cycle loss')
parser.add_argument('--logfile', default='',type=str)
parser.add_argument('--save_epoch',default=-1,type=int,help="frequency of saving images and models")
args = parser.parse_args()


def log_write(text):
    print(text)
    if args.logfile != "":
        with open(f"log/{args.logfile}", "a") as f:
            f.write(text + "\n")

def save_test_image(crt_epoch):
    imgs = next(iter(test_dataloader))
    real_A = imgs["A"].to(device)
    real_B = imgs["B"].to(device)
    fake_B = G(real_A)
    fake_A = F(real_B)
    cyc_A = F(fake_B)
    cyc_B = G(fake_A)

    real_A = make_grid(real_A, normalize=True)
    real_B = make_grid(real_B, normalize=True)
    fake_A = make_grid(fake_A, normalize=True)
    fake_B = make_grid(fake_B, normalize=True)
    cyc_A = make_grid(cyc_A, normalize=True)
    cyc_B = make_grid(cyc_B, normalize=True)
    
    image_grid = torch.stack((real_A, fake_B, cyc_A, real_B, fake_A, cyc_B))
    image_grid = make_grid(image_grid, nrow=3)
    
    save_image(image_grid, f"images/{args.dataset}_epoch{crt_epoch}.png")

def init_weights(m):
    if m.__class__.__name__.find("Conv") != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0.0)

#networks
G = Generator(args.image_size) #A -> B
F = Generator(args.image_size) #B -> A 
D_A = Discriminator(args.image_size)
D_B = Discriminator(args.image_size)

#init parameters
G.apply(init_weights)
F.apply(init_weights)
D_A.apply(init_weights)
D_B.apply(init_weights)

#criterions
criterion_GAN = nn.MSELoss() 
criterion_cycle = nn.L1Loss()
criterion_id = nn.L1Loss()

if cuda:
    G.cuda()
    F.cuda()
    D_A.cuda()
    D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_id.cuda()

G.load_state_dict(torch.load("saved_model/horse2zebra/G/G_050.prm"))
F.load_state_dict(torch.load("saved_model/horse2zebra/F/F_050.prm"))
D_A.load_state_dict(torch.load("saved_model/horse2zebra/D_A/D_A_050.prm"))
D_B.load_state_dict(torch.load("saved_model/horse2zebra/D_B/D_B_050.prm"))

#optimizer
optimizer_G = torch.optim.Adam(chain(G.parameters(), F.parameters()), lr=args.learning_rate)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.learning_rate)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.learning_rate)


#dataloader
train_dataloader = DataLoader(MyDataset(dataname=args.dataset), batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
test_dataloader = DataLoader(MyDataset(dataname=args.dataset, phase='test'))

start_time = time.time()
print("Starting Training Loop...")
save_test_image(0)
for epoch in range(args.n_epochs):
    loss_hist = {"gan":0, "cyc":0, "id": 0, "D":0}
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
        loss_id = criterion_id(real_A, F(real_A)) + criterion_cycle(real_B, G(fake_B))
        loss_gen = loss_GAN + loss_cycle * args.cyc_lambda + loss_id * args.cyc_lambda * 0.5
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

        loss_hist["gan"] += loss_GAN.item() * args.batch_size / 100
        loss_hist["cyc"] += loss_cycle.item() * args.batch_size / 100
        loss_hist["id"] += loss_id.item() * args.batch_size / 100
        loss_hist["D"] += (loss_D_A + loss_D_B).item() * args.batch_size / 200

        #training log
        if (i+1)*args.batch_size % 100 == 0:
            D_real = D_B(real_B).mean().item()
            D_fake = D_B(fake_B).mean().item()
            elapsed_time = int(time.time() - start_time)
            log_write('[%d/%d][%d/%d]  Loss_GAN: %.5f  Loss_cyc: %.5f Loss_id: %.5f  Loss_D: %.5f  D(y): %.5f  D(G(x)): %.5f  time: %d(s)'
                        % (epoch+1, args.n_epochs, (i+1)*args.batch_size, len(train_dataloader)*args.batch_size, *loss_hist.values(), D_real, D_fake, elapsed_time))
            
            loss_hist = {"gan":0, "cyc":0, "id": 0, "D":0}

    if args.save_epoch > 0 and (epoch+1) % args.save_epoch == 0:
        save_test_image(epoch+1)
        torch.save(G.state_dict(), "saved_model/{}/G/G_{:03d}.prm".format(args.dataset, epoch+1))
        torch.save(F.state_dict(), "saved_model/{}/F/F_{:03d}.prm".format(args.dataset, epoch+1))
        torch.save(D_A.state_dict(), "saved_model/{}/D_A/D_A_{:03d}.prm".format(args.dataset, epoch+1))
        torch.save(D_B.state_dict(), "saved_model/{}/D_B/D_B_{:03d}.prm".format(args.dataset, epoch+1))
