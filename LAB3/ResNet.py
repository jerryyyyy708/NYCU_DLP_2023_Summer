import torch
import torch.nn as nn

class ResBlock18(nn.Module):
    def __init__(self, input_size, output_size, stride = 1, downsample = False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(output_size)
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(output_size)
            )
        else:
            self.downsample = None
        
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample is not None:
            #print(x.shape)
            residual = self.downsample(x)
            #print(residual.shape)
        else:
            residual = x
        x = self.conv_layers(x)
        x = x + residual
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)
        self.conv2_x = nn.Sequential(
            ResBlock18(64, 64, 1, False),
            ResBlock18(64, 64, 1, False)
        )
        self.conv3_x = nn.Sequential(
            ResBlock18(64, 128, 2, True),
            ResBlock18(128, 128, 1, False)
        )
        self.conv4_x = nn.Sequential(
            ResBlock18(128, 256, 2, True),
            ResBlock18(256, 256, 1, False)
        )
        self.conv5_x = nn.Sequential(
            ResBlock18(256, 512, 2, True),
            ResBlock18(512, 512, 1, False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class ResBlock50(nn.Module):
    def __init__(self, input_size, first_size, output_size, stride = 1, downsample = False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_size, first_size, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(first_size),
            nn.Conv2d(first_size, first_size, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(first_size),
            nn.Conv2d(first_size, output_size, kernel_size = 1, stride = 1, padding = 0, bias = False)
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(output_size)
            )
        else:
            self.downsample = None
        
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.conv_layers(x)
        x = x + residual
        x = self.relu(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)
        self.conv2_x = nn.Sequential(
            ResBlock50(64, 64, 256, 1, True),
            ResBlock50(256, 64, 256, 1, False),
            ResBlock50(256, 64, 256, 1, False)
        )
        self.conv3_x = nn.Sequential(
            ResBlock50(256, 128, 512, 2, True),
            ResBlock50(512, 128, 512, 1, False),
            ResBlock50(512, 128, 512, 1, False),
            ResBlock50(512, 128, 512, 1, False)
        )
        self.conv4_x = nn.Sequential(
            ResBlock50(512, 256, 1024, 2, True),
            ResBlock50(1024, 256, 1024, 1, False),
            ResBlock50(1024, 256, 1024, 1, False),
            ResBlock50(1024, 256, 1024, 1, False),
            ResBlock50(1024, 256, 1024, 1, False),
            ResBlock50(1024, 256, 1024, 1, False)
        )
        self.conv5_x = nn.Sequential(
            ResBlock50(1024, 512, 2048, 2, True),
            ResBlock50(2048, 512, 2048, 1, False),
            ResBlock50(2048, 512, 2048, 1, False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet152(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)
        conv2_block = []
        conv2_block.append(ResBlock50(64, 64, 256, 1, True))
        for i in range(2):
            conv2_block.append(ResBlock50(256, 64, 256, 1, False))
        self.conv2_x = nn.Sequential(*conv2_block)
        
        conv3_block = []
        conv3_block.append(ResBlock50(256, 128, 512, 2, True))
        for i in range(7):
            conv3_block.append(ResBlock50(512, 128, 512, 1, False))
        self.conv3_x = nn.Sequential(*conv3_block)

        conv4_block = []
        conv4_block.append(ResBlock50(512, 256, 1024, 2, True))
        for i in range(35):
            conv4_block.append(ResBlock50(1024, 256, 1024, 1, False))
        self.conv4_x = nn.Sequential(*conv4_block)
        
        conv5_block = []
        conv5_block.append(ResBlock50(1024, 512, 2048, 2, True))
        for i in range(2):
            conv5_block.append(ResBlock50(2048, 512, 2048, 1, False))
        self.conv5_x = nn.Sequential(*conv5_block)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x