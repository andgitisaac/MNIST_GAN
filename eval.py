import os
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch.nn.functional as F

from config import parse_eval_args
from networks.generator import Generator

args = parse_eval_args()
print(args)

GAN_TYPE = "GAN" if not args.AUX_CLASSIFIER else "ACGAN"
MODEL_DIR = os.path.join("models", GAN_TYPE)
MODEL_PATH = os.path.join(MODEL_DIR, "G_epoch_{:03d}.pth".format(args.EPOCH))
OUTPUT_DIR = os.path.join("outputs", GAN_TYPE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

cudnn.benchmark = True
if args.USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

random.seed(args.SEED)
torch.manual_seed(args.SEED)
if args.USE_CUDA:
    torch.cuda.manual_seed_all(args.SEED)


generator = Generator(args.AUX_CLASSIFIER, args.ZDIM, args.NUM_CLASSES)
generator.load_state_dict(torch.load(MODEL_PATH))
generator.to(device)
generator.eval()


fixedNoise = torch.FloatTensor(args.BATCH_SIZE, args.ZDIM, 1, 1).normal_(0, 1)
if args.AUX_CLASSIFIER:
    fixedClass = F.one_hot(torch.LongTensor([i % args.NUM_CLASSES for i in range(args.BATCH_SIZE)]), num_classes=args.NUM_CLASSES)
    fixedConstraint = fixedClass.unsqueeze(-1).unsqueeze(-1)
    fixed_z = torch.cat((fixedNoise, fixedConstraint), 1)
else:
    fixed_z = fixedNoise
fixed_z.to(device)

            
fakeImage = generator(fixed_z)
vutils.save_image(fakeImage.data,
        "{}/outputs_epoch_{:03d}.png".format(OUTPUT_DIR, args.EPOCH),
        nrow=10,
        normalize=True
)