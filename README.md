# Receipt Count Prediction

A simple machine learning model and app built to predict the approximate number of the scanned receipts that Fetch Rewards receives for each month of 2022 based on data from 2021. The model is developed using PyTorch and the web app is developed using Python and Flask

## Project Details

### Data

-   The training, data data_daily.csv, is provided by Fetch. The data has 1 feature variable "# Date" and a dependent variable "Receipt_Count".
-   There are 365 data points representing number of the observed scanned receipts each day for the year 2021
-   Here is a visualization of the data:
    ![Display Image](https://github.com/andyyhy/receipt_count_prediction/blob/main/results/data_daily_vis.png)

### The Model

-   **Architecture**: The model is a linear regression implemented using PyTorch's `nn.Module`.
-   **Input**: The model takes in the day of the year as input.
-   **Output**: The output is the normalized predicted receipt count for that day.

## Training

The training process involves the following steps:

1. Loading the dataset from `data/data_daily.csv`.
2. Normalizing the data.
3. Splitting the dataset into training and testing sets.
4. Converting the data into PyTorch tensors.
5. Training the model for 1000 epochs.
6. Saving the trained model parameters in `model/model.pth`.

Visualization of the model prediction and the training set data:
![Display Image](https://github.com/andyyhy/receipt_count_prediction/blob/main/results/training_data.png)

## Validation

After training, the model is validated on the test set. The validation loss is computed, and the results are plotted showing both the actual and predicted receipt counts over time.

Visualization of the model prediction and the test set data:
![Display Image](https://github.com/andyyhy/receipt_count_prediction/blob/main/results/test_data.png)

## Results

Below are the predicted approximate number of scanned receipts for each month of 2022:
| Month | # of Receipts |
|------------|--------------|
| 2022-01 | 316,690,180 |
| 2022-02 | 291,861,020 |
| 2022-03 | 329,573,470 |
| 2022-04 | 325,387,260 |
| 2022-05 | 342,893,500 |
| 2022-06 | 338,277,600 |
| 2022-07 | 356,213,500 |
| 2022-08 | 362,982,720 |
| 2022-09 | 357,718,750 |
| 2022-10 | 376,302,720 |
| 2022-11 | 370,609,100 |
| 2022-12 | 389,622,750 |

## Running The Application

![Display Image](https://github.com/andyyhy/receipt_count_prediction/blob/main/results/app.png)
The application will be accessible at http://localhost:8000 in your web browser.

### Running Locally

Make sure you have Python 3.11+ installed on your system. You can install the necessary dependencies by running:
`pip install -r requirements.txt`

To start the web application, navigate to the project directory and run:
`python app/app.py`

### Running With Dockerfile Locally:

Make sure you have docker installed

-   Docker build: `docker build -t username/imagename:v1.0 .`
-   Docker run: `docker container run -d -p 8000:8000 -p 8001:8001 username/imagename:v1.0`

### Running With Docker Hub:

Make sure you have docker installed

-   Docker Pull: `docker pull andyyhy/receipt_count_prediction:latest`
-   Docker run: `docker container run -d -p 8000:8000 -p 8001:8001 andyyhy/receipt_count_prediction:latest`

### Usage

Once the application is running:

1. Navigate to the homepage.
2. Enter a start and end date for the range days that you want to predict
3. Click the "Submit" button.
4. The prediction results will be displayed on the page.

## Project Structure

-   `app/`
    -   `static/`
        -   `data/`
            -   `data_daily.csv`: The dataset for the daily receipt counts.
        -   `js/`
            -   `plot.js`: The dataset for the daily receipt counts.
        -   `styles.css/`: CSS for app.
    -   `templates/`
        -   `index.html`: The HTML template for the user input form.
    -   `app.py`: The main Flask application file with routes and model inference logic.
-   `data/data_daily.csv`: The dataset used for training and testing the model.
-   `model/model.pth`: The trained model parameters saved for later use and/or further evaluation.
-   `model/train_model.py`: The main Python script that defines the model, loads data, trains the model, and validates its performance.s
-   `results`: Folder containing visualizations
