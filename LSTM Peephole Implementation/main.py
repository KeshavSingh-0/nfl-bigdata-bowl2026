# This is literally someone else's implementation. 
# I am just using this to what an implementation would look like.
# https://www.kaggle.com/code/werwar23414141231/lstm-with-peephole/notebook

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# pip install -U imbalanced-learn
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import sys
args = sys.argv
for dirname, _, filenames in os.walk('../kaggle_file/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('../kaggle_file/train_input/input_2023_w01.csv')
df.describe()

########################################################################################
#################### END NOTEBOOK ######################################################
########################################################################################

if args.__contains__("-g"):
    plt.rcParams['font.size'] = 4
    fig, ax = plt.subplots(5, 5, figsize=(9, 4))
    plt.subplots_adjust(wspace=1, hspace=1)

    row = 0
    col = 0
    for value in df.columns:
        sns.histplot(df[value].values, ax=ax[row][col])
        ax[row][col].set_title(value)
        ax[row][col].set_xlim([min(df[value].values), max(df[value].values)])
        col += 1
        if col > 4:
            col = 0
            row += 1
    sns.histplot(df['player_weight'].values, ax=ax[0][0])
    ax[0][0].set_title('Player Weight')
    ax[0][0].set_xlim([min(df['player_weight'].values), max(df['player_weight'].values)])

    sns.histplot(df['player_height'].values, ax=ax[1][0], color='b')
    ax[1][0].set_title('Player Height)')
    ax[1][0].set_xlim([min(df['player_height'].values), max(df['player_height'].values)])

    sns.histplot(df['game_id'].values, ax=ax[2][0], color='b')
    ax[2][0].set_title('Game ID\'s')
    ax[2][0].set_xlim([min(df['game_id'].values) + 299, min(df['game_id'].values)+313])

    print(df)

    plt.show()

########################################################################################
#################### END NOTEBOOK ######################################################
########################################################################################

if args.__contains__("-c"):
    sns.set(rc={'figure.figsize':(15,10)})
    numerical_df = df.select_dtypes(include=np.number)
    ax = sns.heatmap(numerical_df.corr())
    ax.set_title('Correlation Matrix', fontsize=14)
    plt.show()

########################################################################################
#################### END NOTEBOOK ######################################################
########################################################################################

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve

class CustomLSTM_With_Peephole(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.W_f = nn.Parameter(
            torch.Tensor(hidden_sz, hidden_sz))  # switched wf an uf creation (hidden hidde) instead of (input, hidden)
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:  # Weight matrix
                nn.init.xavier_normal_(weight)
            else:  # Bias vector
                nn.init.zeros_(weight)

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.shape
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.W_f + self.b_f + self.V_f * c_t)
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.W_i + self.b_i + self.V_i * c_t)
            c_t = f_t * c_t + i_t * torch.tanh(x_t @ self.U_c + h_t @ self.W_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.W_o + self.b_o + self.V_o * c_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class Net(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super(Net, self).__init__()
        self.layer_1 = CustomLSTM_With_Peephole(input_size, hidden_size)  # input_size, hidden_size
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_2 = CustomLSTM_With_Peephole(hidden_size, 32)  # input_size, hidden_size
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_3 = nn.Linear(32, 1)  # input_size, output_size

    def forward(self, x):
        out, hidden = self.layer_1(x)  # returns tuple consisting of output and sequence
        out = self.dropout1(out)
        out, hidden = self.layer_2(out)
        out = self.dropout2(out)
        out = self.layer_3(hidden[1])
        output = torch.sigmoid(out)
        return output


class Net2(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super(Net2, self).__init__()
        self.layer_1 = CustomLSTM_With_Peephole(input_size, hidden_size)  # input_size, hidden_size
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_2 = CustomLSTM_With_Peephole(hidden_size, 32)  # input_size, hidden_size
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 20)
        self.fc3 = nn.Linear(20, 24)
        self.fc4 = nn.Linear(24, 1)


    def forward(self, x):
        out, hidden = self.layer_1(x)  # returns tuple consisting of output and sequence
        out = self.dropout1(out)
        out, hidden = self.layer_2(out)
        out = torch.relu(self.fc1(hidden[1]))
        out = self.dropout2(out)
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        output = torch.sigmoid(self.fc4(out))
        return output

class Net3(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, 32, kernel_size=3, padding=1)
        self.layer_1 = CustomLSTM_With_Peephole(32, hidden_size)  # input_size now refers to output channels from Conv layer
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_2 = CustomLSTM_With_Peephole(hidden_size, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 20)
        self.fc3 = nn.Linear(20, 24)
        self.fc4 = nn.Linear(24, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Rearrange input to (batch_sz, features, sequence_sz)
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = out.permute(0, 2, 1)  # Rearrange output back to (batch_sz, sequence_sz, features)
        out, hidden = self.layer_1(out)  # returns tuple consisting of output and sequence
        out = self.dropout1(out)
        out, hidden = self.layer_2(out)
        out = torch.relu(self.fc1(hidden[1]))
        out = self.dropout2(out)
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        output = torch.sigmoid(self.fc4(out))
        return output

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data)
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index].float()
        y = self.targets[index].float()
        return x, y


def reshape_data(df):
    num_samples = len(df)
    num_features = len(input_cols)
    num_batches = num_samples // sequence_size * sequence_size
    class_1_df = df[df['Class'] == 1]
    remaining_df = df[df['Class'] != 1]
    df_size = num_batches - len(class_1_df)
    sampled_remaining_df = remaining_df.sample(n=df_size)
    df = pd.concat([class_1_df, sampled_remaining_df])

    data_tensor = df[input_cols].values.reshape(-1, sequence_size, num_features)
    target_tensor = df[target_col].values.reshape(-1)

    return data_tensor, target_tensor


# Load the dataframe
df = df
input_cols = df.columns.drop(['Class'])
target_col = 'Class'
# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
# Define some hyperparameters
sequence_size = 30
n_features = 30  # number of input features
n_hidden = 64  # number of hidden units
batch_size = 128  # size of mini-batches
n_epochs = 10 # number of training epochs
learning_rate = 1e-3  # learning rate for optimizer

# Reshape the training and testing data
train_data_tensor, train_target_tensor = reshape_data(train_df)
test_data_tensor, test_target_tensor = reshape_data(test_df)
# Create custom datasets and dataloaders
train_dataset = CustomDataset(train_data_tensor, train_target_tensor)
test_dataset = CustomDataset(test_data_tensor, test_target_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create an instance of the model
# model = Net2(30,2048,0.25)
# model = Net(30, 2048,0.5)
model = Net3(30,2048,0.5)
# Move the model to the device (CPU or GPU)
model = model.to(device)

# Define the loss function and the optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss for classification
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Adam optimizer

train_losses = []
val_losses = []
train_f1_scores = []
val_f1_scores = []
train_accuracies = []
val_accuracies = []
train_auprcs = []
val_auprcs = []

for epoch in range(n_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_preds, train_targets = [], []
    val_preds, val_targets = [], []
    train_outputs, val_outputs = [], []  # Added: lists to store the raw outputs

    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        train_preds.extend(torch.round(outputs.detach()).cpu().numpy().tolist())
        train_targets.extend(labels.cpu().numpy().tolist())
        train_outputs.extend(outputs.detach().cpu().numpy().tolist())  # Store the raw outputs
        loss = criterion(torch.squeeze(outputs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    model.eval()
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        val_preds.extend(torch.round(outputs.detach()).cpu().numpy().tolist())
        val_targets.extend(labels.cpu().numpy().tolist())
        val_outputs.extend(outputs.detach().cpu().numpy().tolist())  # Store the raw outputs
        loss = criterion(torch.squeeze(outputs), labels)
        val_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_f1 = f1_score(train_targets, train_preds, average='weighted')
    val_f1 = f1_score(val_targets, val_preds, average='weighted')
    train_acc = accuracy_score(train_targets, train_preds)
    val_acc = accuracy_score(val_targets, val_preds)
    train_prec, train_recall, _ = precision_recall_curve(train_targets, train_outputs)  # Calculate precision and recall
    val_prec, val_recall, _ = precision_recall_curve(val_targets, val_outputs)  # Calculate precision and recall
    train_auprc = auc(train_recall, train_prec)  # Calculate AUPRC
    val_auprc = auc(val_recall, val_prec)  # Calculate AUPRC
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_auprcs.append(train_auprc)  # Append AUPRC
    val_auprcs.append(val_auprc)  # Append AUPRC
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train F1 {train_f1:.4f}, Train Acc: {train_acc:.4f}, Train AUPRC: {train_auprc:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Val AUPRC: {val_auprc:.4f}")

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the F1-scores, accuracy, and AUPRC
plt.figure(figsize=(10, 5))
plt.plot(train_f1_scores, label='Train F1-score')
plt.plot(val_f1_scores, label='Validation F1-score')
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(train_auprcs, label='Train AUPRC')  # Plot AUPRC
plt.plot(val_auprcs, label='Validation AUPRC')  # Plot AUPRC
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()