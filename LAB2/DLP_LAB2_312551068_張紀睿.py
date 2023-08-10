import torch
import numpy as np
import torch.nn as nn
from dataloader import *
from model import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from train import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    set_model_seed(seed)
    set_train_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed = 12
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda:0" 
    else:
        device = "cpu" 


    train_x, train_y, test_x, test_y= read_bci_data()

    trainset = BCI_Data(train_x,train_y)
    testset = BCI_Data(test_x,test_y)

    #train the best net
    model = EEGNet("ReLU").float().to(device)

    batch = 256
    lr = 1e-3
    epoch = 400
    show_epoch = 100
    optimizer = "Adam"

    print("Training Best Performance model...")
    train_result, test_result, best_acc, best_epoch = train(model, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr, epochs = epoch, show_epoch = show_epoch)
    print(f"Best test accuracy {best_acc} in epoch {best_epoch}.")

    #Training EEGNet and DepthConvNet
    optimizer = "Adam"
    batch = 256
    lr = 1e-3
    epochs = 400
    show_epoch = 10000
    EEG_ELU_train, EEG_ELU_test, EEG_ELU_best, EEG_ReLU_train, EEG_ReLU_test, EEG_ReLU_best, EEG_LReLU_train, EEG_LReLU_test, EEG_LReLU_best, DCN_ELU_train, DCN_ELU_test, DCN_ELU_best, DCN_ReLU_train, DCN_ReLU_test, DCN_ReLU_best, DCN_LReLU_train, DCN_LReLU_test, DCN_LReLU_best = train_all(trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr, epochs = epochs, show_epoch = show_epoch)

    #Training RNN
    input_size = 2
    hidden_size = 64
    num_layers = 2
    model = RNN(input_size, hidden_size, num_layers).float().to(device)

    batch = 256
    lr = 1e-3
    epoch = 100
    show_epoch = 1000
    optimizer = "Adam"

    print("Training RNN model...")
    train_result, test_result, best_acc, best_epoch = train(model, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr, epochs = epoch, show_epoch = show_epoch, RNN = True)
    print(f"Best test accuracy {best_acc} in epoch {best_epoch}.")

if __name__ == "__main__":
    main()