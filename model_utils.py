import math
import os
import sys
import numpy as np
import psutil
import torch
import torch.utils.data
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr


def check_input(input_text):
    """
    Check for Reset or Exit inputs, respond appropriately
    :param input_text: Text to check
    """
    if input_text == 'r'.casefold():
        # If user inputs 'r'
        try:
            # Close connections, restart program
            p = psutil.Process(os.getpid())
            for handler in p.connections():
                os.close(handler.fd)
        except Exception as e:
            print(e)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    elif input_text == 'x'.casefold():
        # If user input 'x', exit program
        print("Exiting, Farewell...")
        sys.exit(0)


def prompt_login():
    """
    Prompt user for Model Login Password and Stock information,
    If inputs are valid, build stock dataframe, cull extra data,
    calculate a rolling mean per 10-day window, scale using MinMaxScaler.
    :return: stock name, price dataframe, close price reference frame, MinMaxScaler object
    """
    pw = 'Luthor'.casefold()
    login = ''
    print("***************************************************************")
    print("Welcome to the LexCorp Investments Stock Price Prediction Model")
    print("***************************************************************")
    while True:
        try:
            while login != pw:
                login = input("Enter Your LexCorp Password: ")
            # Prompt user to input stock symbol, start, and end dates
            print("Enter 'x' in any input to exit, or 'r' to start over.")
            stock = input("Enter Stock symbol you wish to predict: ")
            check_input(stock)
            start_date = input("Enter a Start Date in YYYY-mm-dd format: ")
            check_input(start_date)
            end_date = input("Enter a Start Date in YYYY-mm-dd format: ")
            check_input(end_date)
            stock_dates = pd.bdate_range(start_date, end_date, freq='B')
            # Instantiate dates dataframe for dates input
            date_frame = pd.DataFrame(index=stock_dates)
            # Get stock data using yahoo finance API
            chosen_stock = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
            break
        except ValueError:
            print("Invalid Input(s), Try Again.")

    # Calculate rolling average per 10-day-window (2-market weeks)
    # Make two copies of (Date, Adj Close) transformed data frame
    reference_frame = chosen_stock[['Close']].copy()
    stock_frame = chosen_stock[['Adj Close']].copy()
    stock_frame['Adj Close'] = stock_frame['Adj Close'].rolling(window=10).mean()
    # Join Stock and Reference Frames with copies of the date frame
    stock_frame = stock_frame.join(date_frame.copy())
    reference_frame = reference_frame.join(date_frame)
    # Use Min/Max Scaler to normalize data to values between -1 and 1
    mm_scale = MinMaxScaler(feature_range=(-1, 1))
    stock_frame['Adj Close'] = mm_scale.fit_transform(stock_frame['Adj Close'].values.reshape(-1, 1))

    return stock, stock_frame, reference_frame, mm_scale


def split_data(stock_data, sequence_length):
    """
    Take in one set of stock prices in data frame format,
    convert into an array of data, split data into two slices equal to 80%/20%.
    :param stock_data: 'Adj Close' prices data frame.
    :param sequence_length: Length of tuples for each array
    :return: [price training data, date training data, price testing data, date testing data]
    """
    data_array = []
    stock_data = stock_data.dropna()
    for i in range(len(stock_data) - sequence_length):
        data_array.append(stock_data[i: i + sequence_length])

    data_array = np.array(data_array)

    # Set testing data size to 20% of our data
    testing_data_size = int(np.round(0.20 * data_array.shape[0]))
    # Set training data size to 75% of our data (100% - testing_data_size)
    training_data_size = data_array.shape[0] - testing_data_size

    # Training Data (First 80% of data-points)
    x_train_split = data_array[:training_data_size, :-1, :]
    y_train_split = data_array[:training_data_size, -1, :]

    # Training Data (Last 20% of data-points)
    x_test_split = data_array[training_data_size:, :-1]
    y_test_split = data_array[training_data_size:, -1, :]

    return [x_train_split, y_train_split, x_test_split, y_test_split]


def train_model(model, x_training_tensor, y_training_tensor, epochs, learning_rate=.01):
    """
    Train a neural network object using historical stock price data. using an x & y training tensor
    :param model: Torch LSTM neural network object.
    :param x_training_tensor: Dates training data in tensor format.
    :param y_training_tensor: Price training data in tensor format.
    :param epochs: Iterations of training
    :param learning_rate: Speed with which to train model (.01 is standard)
    :return: Trained Model object, Mean Squared Error loss data
    """
    # For each iteration, calculate loss, use x,y tensor datasets to train RNN
    training_loss = np.zeros(epochs)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for e in range(epochs):
        model_trained = model(x_training_tensor)
        loss = criterion(model_trained, y_training_tensor)
        training_loss[e] = loss.item()

        # Track loss by displaying epoch and MSE loss every 10 iterations
        if e % 10 == 0:
            print("Epoch: {}   --    MSE: {}".format(e, loss.item()))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # Return model trained, loss data for visualization

    return model_trained, training_loss


def visualize_results(stock_name, reference_df, training_loss, model_trained,
                      model_tested, scaled_train_y, scaled_test_y):
    """
    Display three different visualization plots to show:
    1. Plot Training Loss over Time (in Epochs)
    2. Plot Stock Price Predictions and Actual Historical Stock Prices over Dates
    3. Plot Running Log of Training and Testing Root Mean Squared Error
    :param stock_name: Name of stock originally input by user.
    :param reference_df: Historical 'Close' price dataframe
    :param training_loss: MSE Loss gathered during training
    :param model_trained: Training results returned from 'train_model'.
    :param model_tested: Prediction results from testing model.
    :param scaled_train_y: Min/Max inverted stock price training data
    :param scaled_test_y: Min/Max inverted stock price testing data
    """
    # Because we're using regression, loss is calculated using MSE
    # Calculate and show root mean squared error
    train_root_mse = math.sqrt(mean_squared_error(scaled_train_y[:, 0], model_trained[:, 0]))
    test_root_mse = math.sqrt(mean_squared_error(scaled_test_y[:, 0], model_tested[:, 0]))
    print("Graphed results auto-save to: ../CapstoneProject/prediction_plots")
    print("Close Graphs to Continue...")

    # Format graph colors
    plt.rc('axes', facecolor='#243340', edgecolor='silver')
    plt.rc('lines', color='silver')
    # Graph Training Loss (Mean Squared Error)
    fig1 = plt.figure(figsize=(12, 6), facecolor='silver')
    fig1.text(.5, .5, ("Training Root MSE = $%.2f (USD)" % train_root_mse),
              bbox=dict(facecolor='silver', edgecolor='#243340'))
    fig1.text(1, 0, ("Testing Root MSE = $%.2f (USD)" % test_root_mse),
              bbox=dict(facecolor='silver', edgecolor='#243340'))
    ax = fig1.add_subplot()
    ax.plot(training_loss, color='cyan', label="Training Loss (MSE)")

    # Add Labels and Loss Calculation
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error Percentage')
    legend1 = plt.legend(fancybox=True, facecolor="inherit")
    for text in legend1.get_texts():
        plt.setp(text, color='silver')
    plt.savefig("prediction_plots/{}_training_loss.png".format(stock_name), dpi=300)
    plt.show()

    # Graph (Price Predictions vs. Actual Prices) over (Time)
    fig2 = plt.figure(figsize=(18, 6), facecolor='silver')
    fig2.text(.5, .75, ("Testing Root MSE = $%.2f (USD)" % test_root_mse),
              bbox=dict(facecolor='silver', edgecolor='#243340'))
    ax = fig2.add_subplot()
    ax.xaxis_date()
    # avg_dates_index = stock_df[len(stock_df) - len(scaled_test_y):].index
    full_dates_index = reference_df[len(reference_df) - len(scaled_test_y):].index
    ax.plot(full_dates_index, scaled_test_y, color="#9AA0E3", label="{} Price".format(stock_name))
    ax.plot(full_dates_index, model_tested, color="#C4BCA4", label="Predicted Price")

    # Add Labels and Loss Calculation
    ax.set_title("{} Price Prediction".format(stock_name))
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($USD)')
    legend2 = plt.legend(fancybox=True, facecolor="inherit")
    for text in legend2.get_texts():
        plt.setp(text, color='silver')
    plt.savefig("prediction_plots/{}_predictions.png".format(stock_name), dpi=300)
    plt.show()

    csv_prediction_errors = 'csv_prediction_errors.csv'
    with open(csv_prediction_errors, 'a', newline='') as csv_prediction_file:
        fields = ['stock_name', 'train_root_mse', 'test_root_mse']
        writer = csv.DictWriter(csv_prediction_file, fieldnames=fields)
        writer.writerow({'stock_name': stock_name, 'train_root_mse': train_root_mse,
                         'test_root_mse': test_root_mse})

    stock_names = []
    train_roots = []
    test_roots = []
    with open(csv_prediction_errors, newline='') as csv_prediction_file:
        reader = csv.DictReader(csv_prediction_file, fieldnames=fields)
        for row in reader:
            stock_names.append(row['stock_name'])
            train_roots.append(float(row['train_root_mse']))
            test_roots.append(float(row['test_root_mse']))
            # print(row['stock_name'], row['train_root_mse'], row['test_root_mse'])

    # Calculate Running Average for RMSE
    avg_train_rmse = 0
    avg_test_rmse = 0
    for root in train_roots:
        avg_train_rmse += root
    for root in test_roots:
        avg_test_rmse += root
    avg_train_rmse = avg_train_rmse / len(train_roots)
    avg_test_rmse = avg_test_rmse / len(test_roots)

    # Plot all predictions RMSE
    fig3 = plt.figure(facecolor='silver')
    ax = fig3.add_subplot()
    ax.plot(range(0, len(stock_names), 1), train_roots, color="#9AA0E3", label='Training RMSE')
    ax.plot(range(0, len(stock_names), 1), test_roots, color="#C4BCA4", label='Testing RMSE')
    fig3.text(.15, .71, ("Avg Training RMSE: $%.2f (USD)" % avg_train_rmse),
              bbox=dict(facecolor='silver', edgecolor='#243340'))
    fig3.text(.15, .65, ("Avg Testing RMSE: $%.2f (USD)" % avg_test_rmse),
              bbox=dict(facecolor='silver', edgecolor='#243340'))
    ax.set_title('Prediction Plot')
    ax.set_xlabel('Prediction Iteration')
    ax.set_ylabel('Root Mean Squared Error ($USD)')
    legend3 = plt.legend(fancybox=True, facecolor="inherit")
    for text in legend3.get_texts():
        plt.setp(text, color='silver')
    plt.savefig("prediction_plots/csv_prediction_errors.png", dpi=300)
    plt.show()
