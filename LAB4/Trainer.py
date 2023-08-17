import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.anneal_type = args.kl_anneal_type
        self.anneal_cycle = args.kl_anneal_cycle
        self.anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = -1
        if self.anneal_type == "Cyclical":
            self.L = self.frange_cycle_linear(args.num_epoch)
        elif self.anneal_type == "Monotonic":
            end_period = False
            L = self.frange_cycle_linear(args.num_epoch)
            for i in range(1, args.num_epoch):
                if end_period:
                    L[i] = 1.0
                    continue
                if L[i] < L[i-1]:
                    L[i] = 1
                    end_period = True
            self.L = L
        else:
            self.L = np.ones(args.num_epoch) * self.anneal_ratio
        self.beta = self.L[0]
        print("kl_annealing schedule:", self.L)
        
    def update(self):
        # TODO
        self.current_epoch += 1
        self.beta = self.L[self.current_epoch]
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0):
        # TODO
        n_cycle=self.anneal_cycle
        ratio=self.anneal_ratio

        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle

        step = (stop - start)/(period * ratio)
        
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c* period)] = v
                v += step
                i += 1
        return L
        
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        self.mode = 1
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5, 50, 80], gamma=0.1)#[2,5]
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        train_loss_plot = []
        val_loss_plot = []
        tfr_plot = []
        PSNR_plot = []

        for i in range(self.args.num_epoch):
            img_count = 0
            tfr_plot.append(self.tfr)
            train_loss = 0.0
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False #self.tfr > 0 else False #
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img_count += (img.size(0)*img.size(1))
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                train_loss += loss.detach().cpu() * img.size(0) #not sure
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            val_loss, PSNR = self.eval()

            train_loss_plot.append(train_loss/img_count)
            val_loss_plot.append(val_loss)
            PSNR_plot.append(PSNR.detach().cpu())
            
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            self.gen_image(train_loss_plot, val_loss_plot, tfr_plot, PSNR_plot)
            
    def gen_image(self, train_loss_plot, val_loss_plot, tfr_plot, PSNR_plot, PSNR_frame = False):
        if PSNR_frame:
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_plot, label='PSNR')
            plt.xlabel('Frame')
            plt.ylabel('PSNR')
            plt.title(f'Validation PSNR ({self.args.kl_anneal_type})')
            plt.legend()
            plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_PSNR_per_frame.png')
            plt.close()
            return
    
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_plot, marker='o', label='Training Loss')
        plt.plot(val_loss_plot, marker='o', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve({self.args.kl_anneal_type})')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_loss.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.semilogy(train_loss_plot, marker='o', label='Training Loss')
        plt.semilogy(val_loss_plot, marker='o', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Log Loss Curve({self.args.kl_anneal_type})')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_logloss.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(tfr_plot, marker = 'o', label = 'tfr')
        plt.xlabel('Epochs')
        plt.ylabel('tfr')
        plt.title(f'Teacher Forcing Rate')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_tfr.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(PSNR_plot, marker = 'o', label = 'Valid PSNR')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.title(f'Validation PSNR')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_PSNR.png')
        plt.close()
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        val_loss = 0.0
        val_PSNR = 0.0
        img_count = 0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img_count += (img.size(0) * img.size(1))
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, PSNR = self.val_one_step(img, label)
            val_loss += loss.detach().cpu()
            val_PSNR += PSNR
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        return val_loss/img_count, PSNR/img_count

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        out = img[0]

        reconstruction_loss = 0.0
        kl_loss = 0.0
        for i in range(1, self.train_vi_len):
            label_feat = self.label_transformation(label[i])
            if self.mode == 1:
                out = img[i-1] * self.tfr + out * (1 - self.tfr)
            elif adapt_TeacherForcing:
                out = img[i-1]
            human_feat_hat = self.frame_transformation(out)
            
            z, mu, logvar = self.Gaussian_Predictor(human_feat_hat, label_feat)
            
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)
            out = self.Generator(parm)

            reconstruction_loss += self.mse_criterion(out, img[i])
            kl_loss += kl_criterion(mu, logvar, batch_size = self.batch_size)

        beta = torch.tensor(self.kl_annealing.get_beta()).to(self.args.device)
        loss = reconstruction_loss + beta * kl_loss
        
        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()

        return loss
    
    def val_one_step(self, img, label):
        # TODO
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        decoded_frame_list = [img[0].cpu()]
        out = img[0]
        PSNR = 0.0
        PSNR_perframe = []
        reconstruction_loss = 0.0
        kl_loss = 0.0
        for i in range(1, self.val_vi_len):
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)
            
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)  
            out = self.Generator(parm)
            decoded_frame_list.append(out.cpu())
            PSNR_value = Generate_PSNR(out, img[i])
            PSNR += PSNR_value
            PSNR_perframe.append(PSNR_value.detach().cpu())
            reconstruction_loss += self.mse_criterion(out, img[i])
        
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'{self.args.kl_anneal_type}.gif'))
        loss = reconstruction_loss
        self.gen_image(PSNR_perframe,[], [], [], PSNR_frame = True)

        return loss, PSNR
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):

        if self.current_epoch >= self.tfr_sde and self.tfr > 0:
            #self.tfr *= self.tfr_d_step
            self.tfr -= self.tfr_d_step
        self.tfr = max(self.tfr, 0)
    
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=20, loop=0)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
