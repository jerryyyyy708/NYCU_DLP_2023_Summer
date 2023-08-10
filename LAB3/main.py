import torch
import torch.nn as nn
from dataloader import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from ResNet import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import argparse
from train import *

def get_args():
    parser = argparse.ArgumentParser(description = "for testing, use python main.py -r test -m [model path] -t [model type 18/50/152], the testing csv should be placed in the same folder.")
    parser.add_argument("--run", "-r", default = "train", type = str, required = True, help = 'train/test/finetune/evaluate')
    parser.add_argument("--model", "-m", default ="18", type = str, required = False, help = '18/50/152/*.pt')
    parser.add_argument("--task", "-t", default = "18", type = str, required = False, help = 'only when testing: 18/50/152')
    parser.add_argument("--batch", "-b", default = 16, type = int, required = False, help = "batch size")
    parser.add_argument("--epoch", "-e", default = 50, type = int, required = False, help = "training/finetuning epoch")
    parser.add_argument("--lr", "-l", default = 1e-3, type = float, required = False, help = "learning rate")
    return parser.parse_args()

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda:0" 
    else:
        device = "cpu" 
    
    args = get_args()
    model_name, task, batch, epoch, lr = args.model, args.task, args.batch, args.epoch, args.lr
    if args.run == "train":
        trainset = LeukemiaLoader('','train', '18')
        testset = LeukemiaLoader('', 'valid', '18')
        train_result, test_result, best_acc, best_epoch = train(model_name, trainset, testset, batch = batch, device = device, lr = lr, epochs = epoch, show_epoch = 1)
    elif args.run == "test":
        model = torch.load(model_name).to(device)
        test(model, device, task, f'resnet_{task}_test.csv', f'resnet{task}_output.csv')
    elif args.run == "finetune":
        finetune_valid(model_name, batch, device, lr, epoch)
    elif args.run == "evaluate":
        evaluate(model_name, device)
    

