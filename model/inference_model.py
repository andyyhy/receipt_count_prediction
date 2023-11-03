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

day0 = np.datetime64('2021-01-01')

#Load the data_daily dataset
file_path = 'data/data_daily.csv'
daily_data = pd.read_csv(file_path, parse_dates=['# Date'], index_col=None)


#Load the inference dataset
file_path = 'data/sample_inference.csv'
inference_data = pd.read_csv(file_path, parse_dates=['# Date'], index_col=None)

day_num = []
for date in inference_data['# Date']:
    day_num.append((date - day0).days)  


day_num = np.array(day_num)/364

#print(day_num)
X = torch.from_numpy(day_num.astype(np.float32))

X = X.view(X.shape[0], 1)

#Initialize the model
model = linearRegression(1, 1)

#Load the model
model.load_state_dict(torch.load('model/model.pth'))
model.eval()


#Run the model to make predictions
y_max = np.max(daily_data['Receipt_Count'].to_numpy())
with torch.no_grad():  # Disable gradient computation during inference
    y_predict = model(X)
y_predict *= y_max



print(y_max)
#print the number of predicted receipts for each month of 2022
df = pd.DataFrame({
    'Date': pd.date_range(start='1/1/2022', periods=365),  # Replace with your actual year
    'Receipts': np.array(y_predict.squeeze()) 
})

df = df.set_index('Date')

# Resample by month and sum up the receipts
monthly_receipts = df.resample('M').sum()

# Save the DataFrame to a text file, using a tab as the delimiter
print(monthly_receipts)
print(monthly_receipts)
monthly_receipts.to_csv('prediction_for_each_month_of_2022.txt', sep='\t')


#Plot the results
plt.figure(figsize=(15, 7))
plt.scatter(daily_data['# Date'], daily_data['Receipt_Count'], color='#FBA919', label='Daily Receipt Count', zorder=2)
plt.plot(inference_data['# Date'], y_predict, color='#732385', label='Model Predicted Receipt Count on Inference Data', zorder=2)
plt.title('Daily Scanned Receipts over Time')
plt.xlabel('Date')
plt.ylabel('Number of Receipts')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0) 
plt.legend()

#Display the predicted number of receipts scanned for every month of 2022


# Display the plot
plt.show()