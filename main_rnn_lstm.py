import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # data visualization (e.g plot graphs)
from sklearn.model_selection import train_test_split
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from tqdm import ( tqdm )
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import (
    DataLoader,
    TensorDataset
)  # Gives easier dataset managment by creating mini batches etc.
from scipy import stats

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 20
hidden_size = 256
num_layers = 2
num_classes = 7
learning_rate = 0.002
num_epochs = 10
# Outlier detection threshold (adjust as needed)
outlier_threshold = 3

def detect_outliers(data):
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > outlier_threshold
    print(outliers)
    print(len(outliers))
    return outliers

# creating rnn lstm model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmaxx = nn.Softmax(dim=1)

    def forward(self, x):

        # Set initial hidden and cell states
        batch_size = x.size(0)
        hidden_states = torch.zeros(self.num_layers, self.hidden_size).to(device)
        cell_states = torch.zeros(self.num_layers, self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (hidden_states, cell_states))
    
        # Decode the hidden state of the last time step
        out = self.fc(out)
        out = self.softmaxx(out)
        return out
    
torch.manual_seed(41)
model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)

# getting path and reading the csv file for ALL inputs
# path = kagglehub.dataset_download("nccvector/electromyography-emg-dataset")
# dfPath = path + r"/Electro-Myography-EMG-Dataset/extracted_features_and_labeled_dataset(easiest to work with)/emg_all_features_labeled.csv"

dfPath = r"C:/Users/Team 818/Documents/GitHub/senior-research-neural-network/online_dataset_training_values.csv"
df = pd.read_csv(dfPath)

localdfPath = "C:/Users/Team 818/Documents/GitHub/senior-research-neural-network/emg_data_classified.csv"
localdf = pd.read_csv(localdfPath)

# preparing data for training

dataOnline = df.drop("classification", axis=1) # training data
classOnline = df["classification"] # correct classifcations

dataLocal = localdf.drop("classification", axis=1)
classLocal = localdf["classification"]

# splitting up the data set
X_train, X_test, y_train, y_test = train_test_split(dataLocal, classLocal, test_size=0.10, random_state=41)
# _, x_localtest, _, y_localtest = train_test_split(dataLocal, classLocal, test_size=1.00, random_state=41)

outliers = detect_outliers(X_train)
X_train = X_train.loc[~outliers]
y_train = y_train.loc[~outliers]  # Filter labels corresponding to non-outliers

outliers = detect_outliers(X_test)
X_test = X_test.loc[~outliers]
y_test = y_test.loc[~outliers]  # Filter labels corresponding to non-outliers

SC = StandardScaler()
X_train = pd.DataFrame(SC.fit_transform(X_train))
X_test = pd.DataFrame(SC.transform(X_test))
#x_localtest = pd.DataFrame(SC.transform(dataLocal))

X_train = torch.tensor(X_train.values).float().to(device)
X_test = torch.tensor(X_test.values).float().to(device)
#x_localtest = torch.tensor(x_localtest.values).float().to(device)

y_train = torch.LongTensor(y_train.values).to(device)-1
y_test = torch.LongTensor(y_test.values).to(device)-1
#y_localtest = torch.LongTensor(classLocal.values).to(device)-1

# intializing them as torch datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
#localtest_dataset = TensorDataset(x_localtest, y_localtest)

train_data = DataLoader(train_dataset, batch_size=1)
test_data = DataLoader(test_dataset, batch_size=1)
#localtest_data = DataLoader(localtest_dataset, batch_size=1)

# set criterion, optimizer, and learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train():
    losses = []
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        for batch, (data, targets) in enumerate(tqdm(train_data)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            # forward
            scores = model(data)

            loss = criterion(scores, targets)
            if batch%100 == 0:
                losses.append(loss.cpu().detach().numpy())

            # data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

def check_accuracy(dataset):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for batch, (data, targets) in enumerate(tqdm(dataset)):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    return num_correct / num_samples

train()
print(f"Accuracy on online test set: {check_accuracy(test_data)*100:.2f}")
#print(f"Accuracy on local test set: {check_accuracy(localtest_data)*100:.2f}")
