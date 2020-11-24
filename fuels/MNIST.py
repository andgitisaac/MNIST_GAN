import os
import struct
import numpy as np

import torch

from utils.transforms import train_transform, test_transform

CONFIGS = {
    "train": {
        "images": "train-images-idx3-ubyte",
        "labels": "train-labels-idx1-ubyte",
        "transform": train_transform()
    },
    "test": {
        "images": "t10k-images-idx3-ubyte",
        "labels": "t10k-labels-idx1-ubyte",
        "transform": test_transform()
    }
}

class MNIST(torch.utils.data.Dataset):
    def __init__(self, phase="train"):
        super(MNIST, self).__init__()
        
        if phase not in ["train", "test"]:
            raise ValueError("Phase must be 'train' or 'test'")
        
        self.imageFilePath = os.path.join("data", CONFIGS[phase]["images"])
        self.labelFilePath = os.path.join("data", CONFIGS[phase]["labels"])
        self.transform = CONFIGS[phase]["transform"]
        self.images = None
        self.labels = None

        with open(self.imageFilePath, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            self.images = data.reshape((size, nrows, ncols))

        with open(self.labelFilePath, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

        print("=" * 40)
        print("Loaded {}: {} images".format(self, self.labels.shape[0]))
        print("=" * 40)
    
    def __getitem__(self, index):
        image = self.images[index]
        image = self.transform(image)
        label = self.labels[index]

        return (image, label)

    def __len__(self):
        return self.labels.shape[0]
    
    def __str__(self):
        return 'MNIST Dataset'