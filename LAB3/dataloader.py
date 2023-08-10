import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import torch

def getData(mode, model = "18"):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "train_all":
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        df = pd.read_csv('valid.csv')
        path += df['Path'].tolist()
        label += df['label'].tolist()
        return path, label
    
    else:
        df = pd.read_csv(f'resnet_{model}_test.csv')
        path = df['Path'].tolist()
        label = None
        return path, label

class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode, model = "18", aug = True):
        self.root = root
        self.img_name, self.label = getData(mode, model)
        self.mode = mode
        if 'train' in mode and aug:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees = (0,90)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.Normalize((0.0,), (255.0,))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (255.0,))
            ])

        #print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""
        img_path = self.root + self.img_name[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.mode == "test":
            return img
        label = torch.tensor(self.label[index]).float()
        return img, label