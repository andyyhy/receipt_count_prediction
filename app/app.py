import torch
from flask import Flask, request, render_template, send_file, jsonify
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime, timedelta

#Define the model
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
    

#Initialize the model
model = linearRegression(1, 1)

#Load the model
model.load_state_dict(torch.load('./model/model.pth'))
model.eval()



# Preprocess Input
def preprocess_input(input_data):
    # Implement preprocessing on input_data
    day0 = np.datetime64('2021-01-01')

    day_num = []
    for date in input_data:
        day_num.append((date - day0).days)  
    day_num = np.array(day_num)/364

    X = torch.from_numpy(day_num.astype(np.float32))

    X = X.view(X.shape[0], 1)
    return X


# Function for inference
def run_inference(input_data):
    #Get y_max from the training data

    #Load the data_daily dataset
    file_path = './app/static/data/data_daily.csv'
    daily_data = pd.read_csv(file_path, parse_dates=['# Date'], index_col=None)

    train_data_y_max = np.max(daily_data['Receipt_Count'].to_numpy())
    #print(train_data_y_max)
    # Preprocess the input
    processed_data = preprocess_input(input_data)
    # Run the model
    print(processed_data)
    with torch.no_grad():
        print(list(model.parameters()))
        predictions = model(processed_data)

    return predictions*train_data_y_max

def generate_date_range(start_date, end_date, date_format='%Y-%m-%d'):
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    dates = pd.date_range(start_date, end_date)
    return dates

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Render a form for user input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        # Get data from request
        data = request.get_json()
        start_date = data['start_date']
        end_date = data['end_date']
        dates = generate_date_range(start_date, end_date)
        # Read the file into a pandas DataFrame
        try:
            # Generate the data to run inference
            inference_data = dates
            # Run inference
            predictions = run_inference(inference_data)
            inference_dates = [ts.strftime('%Y-%m-%d') for ts in inference_data]

            # Data for the predicted plot
            predicted_data = {
                'x': inference_dates,  # Convert column to list
                'y': np.array(predictions.squeeze()).tolist(),  # Convert column to list
                'type': 'scatter',
                'name': 'Model Predicted Receipt Count',
                'line': {
                    'color': '#732385'
                }
            }

            # Data for the original plot
            given_data = pd.read_csv('./app/static/data/data_daily.csv', parse_dates=['# Date'], index_col=None)
            given_dates = [ts.strftime('%Y-%m-%d') for ts in given_data['# Date']]
            original_data = {
                'x': given_dates,  # Convert column to list
                'y': given_data['Receipt_Count'].to_list(),  # Convert column to list
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Daily Receipt Count',
                'marker': {
                    'color': '#FBA919'
                } 
            }

            #Calculate the predicted number of receipts for each month of the given range
            df = pd.DataFrame({
                'Date': dates,
                'Receipts': np.array(predictions.squeeze()) 
            })
            
            df = df.set_index('Date')

            # Resample by month and sum up the receipts
            df = df.resample('M').sum()
            #print(df)

            df.index = df.index.strftime('%Y-%m')

            df_dict = df.to_dict()
            list_of_months = []
            #print(df_dict["Receipts"])
            #print(len(df_dict["Receipts"].items()))
            for key, value in df_dict["Receipts"].items():
                #print(key, value)
                list_of_months.append((key, value))
            #print(list_of_months)
            
            return jsonify(success=True, predictedData=predicted_data, originalData = original_data, monthlyReceipts=list_of_months)

        except Exception as e:
            return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("8000"), debug=True)
