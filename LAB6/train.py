import torch
import torch.nn as nn
import torchvision
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from dataloader import DiffusionLoader
from tqdm import tqdm

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=128, 
    blocks = [0, 1, 1], channels = [1, 2, 2]): #Default setting as tutorial
        super().__init__()
        first_channel = class_emb_size//4
        down_blocks = ["DownBlock2D" if x == 0 else "AttnDownBlock2D" for x in blocks]
        up_blocks = ["UpBlock2D" if x == 0 else "AttnUpBlock2D" for x in reversed(blocks)]
        channels = [first_channel * x for x in channels]
        self.model = UNet2DModel(
            sample_size = 64,
            in_channels = 3,
            out_channels = 3,
            layers_per_block = 2,
            block_out_channels = (channels), 
            down_block_types=(down_blocks),
            up_block_types=(up_blocks),
        )
        self.model.class_embedding = nn.Linear(num_classes, class_emb_size)

    def forward(self, x, t, label):
        return self.model(x, t, label).sample


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

train_set = DiffusionLoader()
train_loader = DataLoader(train_set, batch_size = 8, shuffle = True)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

model = ClassConditionedUnet(class_emb_size = 512, blocks = [0, 0, 0, 0, 0, 0], channels = [1, 1, 2, 2, 4, 4]).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
cur_epoch = 0
loss_plot = []

load = True

if load:
    checkpoint = torch.load('./ckpt/63.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    cur_epoch = checkpoint['epoch']
    loss_plot = checkpoint['loss_plot']


n_epochs = 200

print('Start_Training')
for epoch in range(cur_epoch+1, n_epochs):
    train_loss = []
    for x, label in tqdm(train_loader):

        x, label = x.to(device), label.to(device)
        label = label.squeeze(1)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        output = model(noisy_x, timesteps, label)

        loss = loss_function(output, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    epoch_loss = sum(train_loss)/len(train_loss)
    loss_plot.append(epoch_loss)
    print(f'Epoch {epoch}: loss = {epoch_loss}')

    if epoch % 3 == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_plot' : loss_plot
                }, f'./ckpt/{epoch}.pth')