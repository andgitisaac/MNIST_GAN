import os
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision.datasets as dset
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import parse_train_args
from fuels.MNIST import MNIST
from networks.generator import Generator
from networks.discriminator import Discriminator
from utils.helper import weights_init
from utils.transforms import train_transform, augmented_train_transform


# Load the training configurations
args = parse_train_args()
print(args)

# Setup the output dirs
LOG_DIR = os.path.join("logs", args.GAN_TYPE)
MODEL_DIR = os.path.join("models", args.GAN_TYPE)
SAMPLE_DIR = os.path.join("samples", args.GAN_TYPE)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Setup CUDA
cudnn.benchmark = True
if args.USE_CUDA:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

random.seed(args.SEED)
torch.manual_seed(args.SEED)
if args.USE_CUDA:
    torch.cuda.manual_seed_all(args.SEED)


# Initialize Generator
generator = Generator(args.GAN_TYPE, args.ZDIM, args.NUM_CLASSES)
generator.apply(weights_init)
generator.to(device)
print(generator)


# Initialize Discriminator
discriminator = Discriminator(args.GAN_TYPE, args.NUM_CLASSES)
discriminator.apply(weights_init)
discriminator.to(device)
print(discriminator)

# Initialize loss function and optimizer
criterionLabel = nn.BCELoss()
criterionClass = nn.CrossEntropyLoss()

optimizerD = Adam(discriminator.parameters(), lr=args.LR, betas=(0.5, 0.999))
optimizerG = Adam(generator.parameters(), lr=args.LR, betas=(0.5, 0.999))

# Prepare the noise for evaluation during training phase
fixedNoise = torch.FloatTensor(args.BATCH_SIZE, args.ZDIM, 1, 1).normal_(0, 1)
if args.GAN_TYPE in ["CGAN", "ACGAN"]:
    fixedClass = F.one_hot(torch.LongTensor([i % args.NUM_CLASSES for i in range(args.BATCH_SIZE)]), num_classes=args.NUM_CLASSES)
    fixedConstraint = fixedClass.unsqueeze(-1).unsqueeze(-1)
    fixed_z = torch.cat((fixedNoise, fixedConstraint), 1)
else:
    fixed_z = fixedNoise
fixed_z = fixed_z.to(device)


# Prepare the data generator with proper data transformation
if args.AUGMENTED:
    trainDataset = MNIST(augmented_train_transform(), "train")
else:
    trainDataset = MNIST(train_transform(), "train")
trainLoader = DataLoader(trainDataset, batch_size=args.BATCH_SIZE, shuffle=args.SHUFFLE, num_workers=args.NUM_WORKERS)

# testDataset = MNIST("test")
# testLoader = DataLoader(testDataset, batch_size=args.BATCH_SIZE, shuffle=args.SHUFFLE, num_workers=args.NUM_WORKERS)

writer = SummaryWriter(LOG_DIR)

steps = 0
try:    
    for epoch in range(args.EPOCHS):
        for step, (realImage, realClass) in enumerate(tqdm(trainLoader)):

            # Discard the last batch of samples.
            if realImage.size()[0] != args.BATCH_SIZE:
                continue
            
            steps += 1
            
            realClass = realClass.type(torch.LongTensor)
            # Soft labels
            realLabel = torch.FloatTensor(args.BATCH_SIZE, 1).uniform_(0.7, 1.0).to(device)
            fakeLabel = torch.FloatTensor(args.BATCH_SIZE, 1).uniform_(0.0, 0.3).to(device)

            # Flip labels
            # realLabelGT, fakeLabelGT = realLabel, fakeLabel
            # if steps != 0 and steps % 2 == 0:
            #     if random.random() < FLIP:
            #         realLabelGT, fakeLabelGT = fakeLabelGT, realLabelGT

            # Prepare a (batch_size * 1 * w * h) tensor indicates class for discriminator of CGAN
            repeatRealClass = None
            if args.GAN_TYPE == "CGAN":
                repeatRealClass = F.one_hot(realClass, num_classes=args.NUM_CLASSES)
                repeatRealClass = repeatRealClass.unsqueeze(-1).unsqueeze(-1)
                repeatRealClass = repeatRealClass.repeat(1, 1, *realImage.size()[-2:])
                repeatRealClass = repeatRealClass.type(torch.FloatTensor)

                repeatRealClass = repeatRealClass.to(device)

            ### Update Discriminator ### 

            # Train with real
            discriminator.zero_grad()
            realImage = realImage.to(device)            
            
            pred = discriminator(realImage, repeatRealClass)
            if args.GAN_TYPE == "ACGAN":
                predLabel, predClass = pred
            else:
                predLabel = pred
            realClass = realClass.to(device)
            realLabel = realLabel.to(device)
            
            # Compute loss of D with real inputs
            lossRealLabelD = criterionLabel(predLabel, realLabel)
            lossRealClassD = criterionClass(predClass, realClass) if args.GAN_TYPE == "ACGAN" else 0
            lossRealD = lossRealLabelD + lossRealClassD

            accRealLabelD = ((realLabel > 0.5) == (predLabel > 0.5)).sum().item() / args.BATCH_SIZE
            if args.GAN_TYPE == "ACGAN":
                accRealClassD = (realClass == torch.max(predClass, 1)[1]).sum().item() / args.BATCH_SIZE

            # Train with fake
            noise = torch.FloatTensor(args.BATCH_SIZE, args.ZDIM, 1, 1).normal_(0, 1)
            noise = noise.to(device)
            if args.GAN_TYPE in ["CGAN", "ACGAN"]:
                constraint = F.one_hot(realClass, num_classes=args.NUM_CLASSES).unsqueeze(-1).unsqueeze(-1)
                z = torch.cat((noise, constraint), 1)
            else:
                z = noise
            z = z.to(device)

            fakeImage = generator(z)

            # if args.GAN_TYPE == "CGAN":
            #     fakeImage = torch.cat((fakeImage, repeatRealClass), 1)

            pred = discriminator(fakeImage.detach(), repeatRealClass)
            if args.GAN_TYPE == "ACGAN":
                predLabel, predClass = pred
            else:
                predLabel = pred

            # Compute loss of D with fake inputs
            lossFakeLabelD = criterionLabel(predLabel, fakeLabel)
            lossFakeClassD = criterionClass(predClass, realClass) if args.GAN_TYPE == "ACGAN" else 0
            lossFakeD = lossFakeLabelD + lossFakeClassD
            
            accFakeLabelD = ((fakeLabel > 0.5) == (predLabel > 0.5)).sum().item() / args.BATCH_SIZE
            if args.GAN_TYPE == "ACGAN":
                accFakeClassD = (realClass == torch.max(predClass, 1)[1]).sum().item() / args.BATCH_SIZE

            # Optimize D
            lossD = (lossRealD + lossFakeD) / 2
            lossD.backward()
        
            optimizerD.step()
            

            ### Update Generator ### 

            generator.zero_grad()
            pred = discriminator(fakeImage, repeatRealClass)
            if args.GAN_TYPE == "ACGAN":
                predLabel, predClass = pred
            else:
                predLabel = pred
            
            # Compute loss of G
            lossLabelG = criterionLabel(predLabel, realLabel)
            lossClassG = criterionClass(predClass, realClass) if args.GAN_TYPE == "ACGAN" else 0
            lossG = lossLabelG + lossClassG
            
            if args.GAN_TYPE == "ACGAN":
                accClassG = (realClass == torch.max(predClass, 1)[1]).sum().item() / args.BATCH_SIZE

            # Optimize G
            lossG.backward()
            optimizerG.step()


            ### Evaluation Phase ###
            if steps != 0 and steps % args.LOG_STEP == 0:
                # print("Epoch: {:03d}, Step: {:04d} => lossD: {:.4f}, lossG: {:.4f}"
                #         .format(epoch, step, lossD.item(), lossG.item()))

                with torch.no_grad():
                    fakeImage = generator(fixed_z)
                vutils.save_image(fakeImage.data,
                        "{}/samples_epoch_{:03d}.png".format(SAMPLE_DIR, epoch),
                        nrow=10,
                        normalize=True
                )

                writer.add_scalars("Loss", {"lossG": lossG.item(), "lossRealD": lossRealD.item(), "lossFakeD": lossFakeD.item()}, steps)
                writer.add_scalars("LabelAcc", {"accRealLabelD": accRealLabelD, "accFakeLabelD": accFakeLabelD}, steps)

                if args.GAN_TYPE == "ACGAN":
                    writer.add_scalars("ClassAcc", {"accClassG": accClassG, "accRealClassD": accRealClassD, "accFakeClassD": accFakeClassD}, steps)

                writer.add_image("FakeImage", vutils.make_grid(fakeImage.data, nrow=10, normalize=True), steps)
        
        ### Save model ###
        torch.save(generator.state_dict(), "{}/G_epoch_{:03d}.pth".format(MODEL_DIR, epoch))
        torch.save(discriminator.state_dict(), "{}/D_epoch_{:03d}.pth".format(MODEL_DIR, epoch))

except KeyboardInterrupt as ke:
    print("Interrupted")
except:
    import traceback
    traceback.print_exc()
finally:
    pass