import torch
import pandas as pd
import matplotlib.pyplot as plt # data visualization (e.g plot graphs)
from sklearn.model_selection import train_test_split
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 80
hidden_size = 256
num_layers = 2
num_classes = 7
learning_rate = 0.002
num_epochs = 1

# creating rnn lstm model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (hidden_states, cell_states))
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out, x
    
torch.manual_seed(41)
model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)

# getting path and reading the csv file for ALL inputs
path = r"C:\Users\gmand\.cache\kagglehub\datasets\nccvector\electromyography-emg-dataset\versions\1\Electro-Myography-EMG-Dataset\extracted_features_and_labeled_dataset(easiest to work with)\emg_all_features_labeled.csv"
df = pd.read_csv(path)

# preparing data for training

x = df.drop("1.2", axis=1) # training data
y = df["1.2"] # correct classifcations

# splitting up the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=41)

x_train = torch.tensor(x_train.values)
x_test = torch.tensor(x_test.values)

y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)

# set criterion, optimizer, and learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training the network

losses = []

def train():
    for epoch in range(num_epochs):
        for data, targets in zip(x_train, y_train):
            # Get data to cuda if possible
            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.detach().numpy())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()
    print(losses)
        
train()