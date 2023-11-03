import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Define the model
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

#Load the dataset
file_path = 'data/data_daily.csv'
daily_data = pd.read_csv(file_path, parse_dates=['# Date'], index_col=None)

#Process the data
X_numpy = np.arange(0, 365)
y_numpy = daily_data['Receipt_Count'].to_numpy()


x_max = np.max(X_numpy)
y_max = np.max(y_numpy)
print(x_max)

X_numpy = X_numpy/np.max(X_numpy)
y_numpy = y_numpy/np.max(y_numpy)

#Slit into Training and Test sets
n_samples = 365
train_size = int(0.8*n_samples)

X_train = X_numpy[:train_size]
X_test = X_numpy[train_size:]


y_train = y_numpy[:train_size]
y_test = y_numpy[train_size:]

date_train = daily_data['# Date'][:train_size]
date_test = daily_data['# Date'][train_size:]

print(date_train)

#Turn into tensors
X = torch.from_numpy(X_train.astype(np.float32))
y = torch.from_numpy(y_train.astype(np.float32))
X = X.view(X.shape[0], 1)
y = y.view(y.shape[0], 1)

#Initialize the model
n_samples, n_features = X.shape
input_size = n_features
output_size = 1
learning_rate = 0.1
model = linearRegression(input_size, output_size)

#Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    # backward pass
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()
    
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#Save the model
torch.save(model.state_dict(), 'model/model.pth')


#Validate the model
#Turn into tensors
X = torch.from_numpy(X_test.astype(np.float32))
y = torch.from_numpy(y_test.astype(np.float32))
X = X.view(X.shape[0], 1)
y = y.view(y.shape[0], 1)

with torch.no_grad():  # Disable gradient calculation
    predicted_y = model(X)

val_loss = criterion(predicted_y, y).item()
print(f'Validation Loss: {val_loss}')

# Plot the test data prediction and the actual data
plt.figure(figsize=(15, 7))
plt.scatter(date_test, y_test*y_max, color='#FBA919', label='Daily Receipt Count', zorder=2)
plt.plot(date_test, predicted_y*y_max, color='#732385', label='Model Predicted Receipt Count on Training Data', zorder=2)
plt.title('Daily Scanned Receipts over Time')
plt.xlabel('Date')
plt.ylabel('Number of Receipts')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0) 

plt.legend()

# Display the plot
plt.show()
