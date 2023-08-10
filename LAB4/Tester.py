import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
from torchvision.utils import save_image
from torch import stack

import imageio
from math import log10
from Trainer import VAE_Model
import glob


TA_ = """
 ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗    ██╗██╗██╗
██╔════╝██╔═══██╗████╗  ██║██╔════╝ ██╔══██╗██╔══██╗╚══██╔══╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝    ██║██║██║
██║     ██║   ██║██╔██╗ ██║██║  ███╗██████╔╝███████║   ██║   ██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗    ██║██║██║
██║     ██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║    ╚═╝╚═╝╚═╝
╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║    ██╗██╗██╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚═╝╚═╝╚═╝                                                                                                                          
"""


def generate_Ground_truth(idx):
    loaded_2D_mat = np.loadtxt(f'Demo_Test/GT{idx}.csv')
    img = torch.from_numpy(loaded_2D_mat).reshape(1, 630 ,3, 32, 64)
    return img

def make_ground_truth_gif(save_root, img_tensor, idx):
    new_list = []
    for i in range(630):
        new_list.append(transforms.ToPILImage()(img_tensor[i]))
        
    new_list[0].save(os.path.join(save_root, f'seq{idx}/ground_truth.gif'), \
        format="GIF", append_images=new_list,
                save_all=True, duration=20, loop=0)

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def calculate_PSNR(save_root, gen_image, idx):
    ground_truth = generate_Ground_truth(idx)[0]
    make_ground_truth_gif(save_root, ground_truth, idx)
    gen_image = gen_image[0]
    
    PSNR_LIST = []
    for i in range(1, 630):
        PSNR = Generate_PSNR(ground_truth[i], gen_image[i])
        PSNR_LIST.append(PSNR.item())
        
    return sum(PSNR_LIST)/(len(PSNR_LIST)-1)
        

def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

from torch.utils.data import DataLoader
from glob import glob
from torch.utils.data import Dataset as torchData
from torchvision.datasets.folder import default_loader as imgloader



class Dataset_Dance(torchData):
    def __init__(self, root, transform, mode='test', video_len=7, partial=1.0):
        super().__init__()
        self.img_folder = []
        self.label_folder = []
        
        data_num = len(glob('./Demo_Test/*'))
        for i in range(data_num):
            self.img_folder.append(sorted(glob(os.path.join(root , f'test/test_img/{i}/*')), key=get_key))
            self.label_folder.append(sorted(glob(os.path.join(root , f'test/test_label/{i}/*')), key=get_key))
        
        self.transform = transform

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, index):
        frame_seq = self.img_folder[index]
        label_seq = self.label_folder[index]
        
        imgs = []
        labels = []
        imgs.append(self.transform(imgloader(frame_seq[0])))
        for idx in range(len(label_seq)):
            labels.append(self.transform(imgloader(label_seq[idx])))
        return stack(imgs), stack(labels)


class Test_model(VAE_Model):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass     
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        psnr_list = []
        for idx, (img, label) in enumerate(tqdm(val_loader)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            psnr = self.val_one_step(img, label, idx)
            psnr_list.append(psnr)
        
        
        fin_text = "============== Your Result ==============\n"
        for i in range(len(psnr_list)):
            fin_text += f"PSNR for testing sequence{i+1}: {psnr_list[i]:.3f}\n"
        avg_psnr = sum(psnr_list) / len(psnr_list)
        fin_text += "-----------------------------------------\n"
        fin_text += f"                       AVG: {avg_psnr:.3f}\n"
        
        print(TA_)
        print(fin_text)
            
    
    def val_one_step(self, img, label, idx=0):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        assert img.shape[0] == 1, "Testing video seqence should be 1"
        
        decoded_frame_list = [img[0].cpu()]
        label_list = []

        # Normal normal
        last_human_feat = self.frame_transformation(img[0])
        first_templete = last_human_feat.clone()
        out = img[0]
        
        for i in range(1, self.val_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)
            
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            out = self.Generator(parm)
            
            decoded_frame_list.append(out.cpu())
            label_list.append(label[i].cpu())
            
        
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        
        os.makedirs(os.path.join(args.save_root, f'seq{idx}'), exist_ok=True)

        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'seq{idx}/pred_seq.gif'))
        self.make_gif(label_frame[0], os.path.join(self.args.save_root, f'seq{idx}/pose.gif'))
        PSNR = calculate_PSNR(self.args.save_root, generated_frame, idx)
        return PSNR
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=20, loop=0)
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, video_len=self.val_vi_len)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 




def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = Test_model(args).to(args.device)
    model.load_checkpoint()
    model.eval()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--no_sanity',     action='store_true')
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--make_gif',      action='store_true')
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
