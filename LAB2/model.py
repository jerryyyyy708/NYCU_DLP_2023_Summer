import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def set_model_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class BCI_Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self,index):
        x = self.x[index]
        y = self.y[index]
        return torch.from_numpy(x).float(), torch.tensor(y).float()
    
    def __len__(self):
        return len(self.y)

class EEGNet(nn.Module):
    def __init__(self, activation = "ELU"):
        super().__init__()
        if activation == "ELU":
            activation_funct = nn.ELU(alpha = 1.0)
        elif activation == "ReLU":
            activation_funct = nn.ReLU()
        else:
            activation_funct = nn.LeakyReLU()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (1,51), stride=(1, 1), padding =(0, 25), bias = False),
            nn.BatchNorm2d(16, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (2, 1), stride=(1, 1), groups = 16, bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            activation_funct,
            nn.AvgPool2d(kernel_size = (1, 4),stride = (1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32,32,  kernel_size = (1,15), stride = (1, 1), padding = (0, 7),bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            activation_funct,
            nn.AvgPool2d(kernel_size = (1, 8), stride = (1, 8), padding = 0),
            nn.Dropout(p = 0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features = 736, out_features = 2, bias = True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, 736)
        x = self.classify(x)
        return x
    
class DeepConvNet(nn.Module):
    def __init__(self, activation = "ELU"):
        super().__init__()
        if activation == "ELU":
            activation_funct = nn.ELU(alpha = 1.0)
        elif activation == "ReLU":
            activation_funct = nn.ReLU()
        else:
            activation_funct = nn.LeakyReLU()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = (1,5), padding = "valid"),
            nn.Conv2d(25,25, kernel_size = (2,1), padding = "valid"),
            nn.BatchNorm2d(25, eps = 1e-05, momentum =0.1),
            activation_funct,
            nn.MaxPool2d(kernel_size = (1,2)),
            nn.Dropout(p = 0.5),
            nn.Conv2d(25,50, kernel_size= (1,5), padding = "valid"),
            nn.BatchNorm2d(50, eps = 1e-05, momentum =0.1),
            activation_funct,
            nn.MaxPool2d(kernel_size = (1,2)),
            nn.Dropout(p = 0.5),
            nn.Conv2d(50,100, kernel_size= (1,5), padding = "valid"),
            nn.BatchNorm2d(100, eps = 1e-05, momentum =0.1),
            activation_funct,
            nn.MaxPool2d(kernel_size = (1,2)),
            nn.Dropout(p = 0.5),
            nn.Conv2d(100,200, kernel_size= (1,5), padding = "valid"),
            nn.BatchNorm2d(200, eps = 1e-05, momentum =0.1),
            activation_funct,
            nn.MaxPool2d(kernel_size = (1,2)),
            nn.Dropout(p = 0.5)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features = 200*43, out_features = 2, bias = True)
        )
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1,200*43)
        x = self.classify(x)
        return x
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.classify = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.rnn(x, h0)
        x = self.classify(x[:, -1, :])
        return x
