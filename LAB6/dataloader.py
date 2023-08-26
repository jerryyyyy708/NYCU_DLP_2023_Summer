from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import json
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class DiffusionLoader(Dataset):
    def __init__(self, root = 'iclevr', json_file = "train.json"):
        self.dict = self.Load_Dict()
        self.root = root
        if 'train' in json_file:
            self.img, self.label = self.Load_Trainset(json_file)
        elif 'test' in json_file:
            self.img = None
            self.label = self.Load_Testset(json_file)
        elif 'both' in json_file:
            test = self.Load_Testset('test.json')
            test2 = self.Load_Testset('new_test.json')
            self.img = None
            self.label = test2 + test
        self.mlb = MultiLabelBinarizer().fit([(x, ) for x in range(24)])
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.img is not None:
            transform = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            img_path = os.path.join(self.root, self.img[index])
            img = transform(Image.open(img_path).convert('RGB'))
            label = self.mlb.transform([self.label[index]])
            return img.float(), torch.from_numpy(label).float()
        else:
            label = self.mlb.transform([self.label[index]])
            return torch.from_numpy(label).float()

    def Load_Dict(self):
        with open('objects.json', 'r') as file:
            data = json.load(file)
        return data

    def Load_Trainset(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            img = []
            label = []
            for i in data:
                img.append(i)
                label_names = data[i]
                labels = [self.dict[x] for x in label_names]
                label.append(np.array(labels))
                
        return img, label
    
    def Load_Testset(self, json_file):
        with open(json_file, 'r') as file:
            label = []
            data = json.load(file)
            for i in data:
                label.append(np.array([self.dict[x] for x in i]))
        return label
    
    
