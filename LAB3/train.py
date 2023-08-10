import torch
import torch.nn as nn
from dataloader import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from ResNet import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from sklearn.metrics import confusion_matrix

def SaveCsvForPlot(filename, train_result, test_result):
    data = [(train_acc, test_acc) for train_acc, test_acc in zip(train_result, test_result)]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def train(model_name, trainset, testset, batch = 64, device = "cuda:0", lr = 1e-2, epochs = 150, show_epoch = 10, train_only = False, save_best = True, save_from = 10000):
    if '.pt' in model_name:
        model = torch.load(model_name).to(device)
        model_name = 'Load'
    elif model_name == "18":
        model = ResNet18().to(device)
    elif model_name == "50":
        model = ResNet50().to(device)
    else:
        model = ResNet152().to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(trainset,batch,shuffle=True)
    if not train_only:
        test_loader = DataLoader(testset,batch,shuffle=True)
    else:
        testset = [0, 0]
    
    best = 0
    bestepoch = 0
    train_accuracy_plot = []
    test_accuracy_plot = []
    
    for epoch in range(epochs):
        model.train()
        train_loss=0
        train_accuracy=0.0
        for x, y in tqdm(train_loader):
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
        if not train_only:
            with torch.no_grad():
                for x, y in tqdm(test_loader):
                    x, y = x.to(device), y.to(device).to(torch.int64)
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
            if save_best:
                torch.save(model,f'ResNet{model_name}_best_{epoch}.pt')
            best = test_accuracy/len(testset)
            bestepoch = epoch
        if epoch+1 >= save_from:
            torch.save(model,f'ResNet{model_name}_epoch{epoch+1}.pt')
        torch.save(model,f'ResNet{model_name}_last.pt')
        if not train_only:
            SaveCsvForPlot(f'ResNet{model_name}_train_log.csv',train_accuracy_plot, test_accuracy_plot)
    
    return train_accuracy_plot, test_accuracy_plot, best, bestepoch

def train_all(batch, device, lr,  epoch, show_epoch = 1, save_best = True, save_from = 10000):
    trainset = LeukemiaLoader('','train')
    testset = LeukemiaLoader('', 'valid')
    train_18, test_18, best_18, _ = train("18", trainset, testset, batch = batch, device = device, lr = lr, epochs = epoch, show_epoch = show_epoch, save_best = save_best, save_from = save_from)
    train_50, test_50, best_50, _ = train("50", trainset, testset, batch = batch, device = device, lr = lr, epochs = epoch, show_epoch = show_epoch, save_best = save_best, save_from = save_from)
    train_152, test_152, best_152, _ = train("152", trainset, testset, batch = batch, device = device, lr = lr, epochs = epoch, show_epoch = show_epoch, save_best = save_best, save_from = save_from)
    print("Best Accuracy: ", best_18, best_50, best_152)
    plt.figure(figsize=(12, 9)) 
    plt.plot(train_18, label = "ResNet18 Train")
    plt.plot(test_18, label = "ResNet18 Test")
    plt.plot(train_50, label = "ResNet50 Train")
    plt.plot(test_50, label = "ResNet50 Test")
    plt.plot(train_152, label = "ResNet152 Train")
    plt.plot(test_152, label = "ResNet152 Test")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.show()


def finetune_valid(model_name, batch, device, lr, epoch = 20, show_epoch = 1, save_from = 0):
    dataset = LeukemiaLoader('','train_all',aug = False)
    train_result, _, _, _ = train(model_name, dataset, None, batch, device, lr, epoch, show_epoch = show_epoch, train_only= True, save_best = False, save_from = save_from)


def test(model, device, model_name = '18', test_file = 'resnet_18_test.csv', filename = "./your_student_id_resnet18.csv"):
    testset = LeukemiaLoader('', 'test', model_name)
    test_loader = DataLoader(testset, 1, shuffle= False)
    pred = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            output = model(x).float()
            if output[0][1] > output[0][0]:
                pred.append(1)
            else:
                pred.append(0)
    save_result(test_file , pred, filename)

def save_result(csv_path, predict_result, filename):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv(filename, index=False)

def evaluate(model_file, device, valid_file = "valid.csv"):
    model = torch.load(model_file).to(device)
    validset = LeukemiaLoader('', 'valid', '')
    valid_loader = DataLoader(validset, 1, shuffle = False)
    pred = []
    ground_truth = []
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x).float()
            pred.append(int(output[0][1] > output[0][0]))
            ground_truth.append(int(y > 0.5))
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(ground_truth, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Model: {model_file[0:-3]}")
    plt.show()