# Receipt Count Prediction

A simple machine learning model and app built to predict the approximate number of the scanned receipts that Fetch Rewards receives for each month of 2022 based on data from 2021. The model is developed using PyTorch and the web app is developed using Python and Flask

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

## Running Locally

Make sure you have Python 3.11+ installed on your system. You can install the necessary dependencies by running:
`pip install -r requirements.txt`

To start the web application, navigate to the project directory and run:
`python app/app.py`
The application will be accessible at http://localhost:8000 in your web browser.

### Data

-   The training, data data_daily.csv, is provided by Fetch. The data has 1 feature variable "# Date" and a dependent variable "Receipt_Count".
-   There are 365 data points representing number of the observed scanned receipts each day for the year 2021
-   Here is a visualization of the data:
    ![Display Image](https://github.com/andyyhy/receipt_count_prediction/blob/main/results/data_daily_vis.png)

### The Model

-
