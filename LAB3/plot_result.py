import pandas as pd
import matplotlib.pyplot as plt

ResNet18_Train = pd.concat([pd.read_csv('./old_models/log/ResNet18_train_log.csv', header = None), pd.read_csv('./old_models/log/ResNet18Load_train_log.csv', header = None)])
ResNet50_Train = pd.concat([pd.read_csv('./old_models/log/ResNet50_train_log.csv', header = None), pd.read_csv('./old_models/log/ResNet50Load_train_log.csv', header = None)])
ResNet152_Train = pd.concat([pd.read_csv('./old_models/log/ResNet152_train_log.csv', header = None), pd.read_csv('./old_models/log/ResNet152Load_train_log.csv', header = None)])
train_column = 0  # Column index for train accuracy
valid_column = 1  # Column index for validation accuracy
ResNet18_Train.reset_index(drop=True, inplace=True)
ResNet50_Train.reset_index(drop=True, inplace=True)
ResNet152_Train.reset_index(drop=True, inplace=True)


# Plotting all models together
plt.figure(figsize=(10, 1.5))

# ResNet18
plt.plot(ResNet18_Train[train_column], label='ResNet18 Train')
plt.plot(ResNet18_Train[valid_column], label='ResNet18 Validation')

# ResNet50
plt.plot(ResNet50_Train[train_column], label='ResNet50 Train')
plt.plot(ResNet50_Train[valid_column], label='ResNet50 Validation')

# ResNet152
plt.plot(ResNet152_Train[train_column], label='ResNet152 Train')
plt.plot(ResNet152_Train[valid_column], label='ResNet152 Validation')

plt.title('Training and Validation Accuracy for ResNet Models')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()