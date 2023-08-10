import torch
import torch.nn as nn
from model import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

def set_train_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda:0" 
    return "cpu" 

def train(model, trainset, testset, optimizer = "Adam", batch = 64, device = "cuda:0", lr = 1e-2, epochs = 150, show_epoch = 10, RNN = False):
    loss_function = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_loader = DataLoader(trainset,batch,shuffle=True)
    test_loader = DataLoader(testset,batch,shuffle=True)
    
    best = 0
    train_accuracy_plot = []
    test_accuracy_plot = []
    
    for epoch in range(epochs):
        model.train()
        train_loss=0
        train_accuracy=0.0
        for x, y in train_loader:
            if RNN:
                x = x.view(-1, 750, 2).to(device) #RNN only
            x, y = x.to(device), y.to(device).to(torch.int64)
            optimizer.zero_grad()
            output=model(x).float()
            loss=loss_function(output,y)
            loss.backward()
            train_loss+=loss
            optimizer.step()
            _,preds = torch.max(output.data,1)
            train_accuracy+=int(torch.sum(preds==y.data))
            
        model.eval()
        test_loss=0
        test_accuracy=0.0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).to(torch.int64)
            if RNN:
                x = x.view(-1, 750, 2).to(device) #for RNN only
            output=model(x).float()
            loss=loss_function(output,y)
            test_loss+=loss
            _,preds = torch.max(output.data,1)
            test_accuracy+=int(torch.sum(preds==y.data))
        if (epoch+1) % show_epoch == 0:
            print(f"Epoch {epoch+1}: train accuracy = {train_accuracy/len(trainset)}, test accuracy = {test_accuracy/len(testset)}")    
        train_accuracy_plot.append(train_accuracy/len(trainset))
        test_accuracy_plot.append(test_accuracy/len(testset))
        if test_accuracy/len(testset) > best:
            best = test_accuracy/len(testset)
            bestepoch = epoch

    return train_accuracy_plot, test_accuracy_plot, best, bestepoch

def train_all(trainset, testset, optimizer = "Adam", batch = 64, device = "cuda:0", lr = 1e-2, epochs = 150, show_epoch = 10000):
    
    print("Training EEG_ELU model...")
    EEG_ELU = EEGNet(activation= "ELU").float().to(device)
    EEG_ELU_train , EEG_ELU_test, EEG_ELU_best, EEG_ELU_bestE = train(EEG_ELU, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr , epochs = epochs,  show_epoch = show_epoch)
    print("Best Accuracy: ", EEG_ELU_best, " on epoch ", EEG_ELU_bestE)

    print("Training EEG_ReLU model...")
    EEG_ReLU = EEGNet(activation= "ReLU").float().to(device)
    EEG_ReLU_train, EEG_ReLU_test, EEG_ReLU_best, EEG_ReLU_bestE = train(EEG_ReLU, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr , epochs = epochs,  show_epoch = show_epoch)
    print("Best Accuracy: ", EEG_ReLU_best," on epoch ", EEG_ReLU_bestE)

    print("Training EEG_Leaky_ReLU model...")
    EEG_LReLU = EEGNet(activation= "Leaky_ReLU").float().to(device)
    EEG_LReLU_train, EEG_LReLU_test, EEG_LReLU_best, EEG_LReLU_bestE = train(EEG_LReLU, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr , epochs = epochs,  show_epoch = show_epoch)
    print("Best Accuracy: ", EEG_LReLU_best, " on epoch ", EEG_LReLU_bestE)

    print("Training DeepConvNet_ELU model...")
    DCN_ELU = DeepConvNet(activation= "ELU").float().to(device)
    DCN_ELU_train , DCN_ELU_test, DCN_ELU_best, DCN_ELU_bestE = train(DCN_ELU, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr , epochs = epochs,  show_epoch = show_epoch)
    print("Best Accuracy: ", DCN_ELU_best," on epoch ", DCN_ELU_bestE)

    print("Training DeepConvNet_ReLU model...")
    DCN_ReLU = DeepConvNet(activation= "ReLU").float().to(device)
    DCN_ReLU_train , DCN_ReLU_test, DCN_ReLU_best, DCN_ReLU_bestE = train(DCN_ReLU, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr , epochs = epochs,  show_epoch = show_epoch)
    print("Best Accuracy: ", DCN_ReLU_best, " on epoch ", DCN_ReLU_bestE)

    print("Training DeepConvNet_LeakyReLU model...")
    DCN_LReLU = DeepConvNet(activation= "LeakyReLU").float().to(device)
    DCN_LReLU_train , DCN_LReLU_test, DCN_LReLU_best, DCN_LReLU_bestE = train(DCN_LReLU, trainset, testset, optimizer = optimizer, batch = batch, device = device, lr = lr , epochs = epochs,  show_epoch = show_epoch)
    print("Best Accuracy: ", DCN_LReLU_best, " on epoch ", DCN_LReLU_bestE)

    return EEG_ELU_train, EEG_ELU_test, EEG_ELU_best, EEG_ReLU_train, EEG_ReLU_test, EEG_ReLU_best, EEG_LReLU_train, EEG_LReLU_test, EEG_LReLU_best, DCN_ELU_train, DCN_ELU_test, DCN_ELU_best, DCN_ReLU_train, DCN_ReLU_test, DCN_ReLU_best, DCN_LReLU_train, DCN_LReLU_test, DCN_LReLU_best