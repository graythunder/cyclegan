import os, sys
from pathlib import Path
from PIL import Image
import random
import torch
import torch.utils.data as data
from torchvision import transforms

class MyDataset(data.Dataset):
    def __init__(self, datadir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"), 
        dataname="zebra2horse", phase="train"):
        super().__init__()
        
        self.dir_path = os.path.join(datadir, dataname)
        
        self.imageA_paths = sorted([str(p) for p in Path(os.path.join(self.dir_path, f"{phase}A")).glob("**/*.*")])
        self.imageB_paths = sorted([str(p) for p in Path(os.path.join(self.dir_path, f"{phase}B")).glob("**/*.*")])

        transform_ops = []
        #TODO: preprocess, augmentation

        # if phase == "train":
        #     transform_ops.extend([
        #         transforms.RandomVerticalFlip(),
        #         transforms.RandomRotation(20),
        #         transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0))
        #     ])
            
        transform_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transformer = transforms.Compose(transform_ops)
        
    def __len__(self):
        return max(len(self.imageA_paths), len(self.imageB_paths))
    
    def __getitem__(self, index):
        pathA = self.imageA_paths[index % len(self.imageA_paths)]
        pathB = self.imageB_paths[random.randint(0, len(self.imageB_paths) - 1)]
        
        imageA = Image.open(pathA)
        imageA = self.transformer(imageA)
        imageB = Image.open(pathB)
        imageB = self.transformer(imageB)

        return {"A": imageA, "B": imageB}