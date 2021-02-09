import sys
import os
import psutil as psutil
import yfinance as yf
import torch.utils.data
from Prediction_Model import NN
from model_utils import prompt_login, split_data, train_model, visualize_results

if __name__ == '__main__':
    # Allow pandas data reader
    yf.pdr_override()
    # Prompt user for Stock Symbol, Start and End Dates
    while True:
        # If inputs error, display message, prompt again
        try:
            stock, stock_frame, reference_frame, min_max_scale = prompt_login()
            break
        except ValueError:
            print("Invalid Input(s), Try Again.")

    print("Building Data Sets...")
    # Set training/testing length, split data into training and testing sets
    set_length = 28
    # X_train, Y_train, X_test, Y_test = split_data(stock_frame, set_length)
    X_train, Y_train, X_test, Y_test = split_data(stock_frame, set_length)
    # print(X_train, Y_train, X_test, Y_test)

    # Convert training/testing sets into tensors
    x_train_tens = torch.from_numpy(X_train).type(torch.Tensor)
    y_train_tens = torch.from_numpy(Y_train).type(torch.Tensor)
    x_test_tens = torch.from_numpy(X_test).type(torch.Tensor)
    y_test_tens = torch.from_numpy(Y_test).type(torch.Tensor)

    # Set values for model training/testing
    input_d = 1
    batch_s = 100
    layers_n = 2
    output_d = 1
    # 100th epoch seems to be around the point when MSE flattens
    epochs = 120

    # Create data loaders using Dataset objects
    train_set = torch.utils.data.TensorDataset(x_train_tens, y_train_tens)
    test_set = torch.utils.data.TensorDataset(x_test_tens, y_test_tens)
    torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_s, shuffle=False)
    torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_s, shuffle=False)

    # Initialize Recurrent Neural Network (LSTM)
    RNN = NN(input_dim=input_d, hidden_dim=batch_s, num_layers=layers_n, output_dim=output_d)

    # Train Model
    print("Training Model...")
    model_train, train_loss = train_model(RNN, x_train_tens, y_train_tens, 0.01, epochs)

    print("Making Predictions...")
    # Back-test Model
    model_test = RNN(x_test_tens)

    # Invert data for visualization
    model_train = min_max_scale.inverse_transform(model_train.detach().numpy())
    model_test = min_max_scale.inverse_transform(model_test.detach().numpy())
    y_train = min_max_scale.inverse_transform(y_train_tens.detach().numpy())
    y_test = min_max_scale.inverse_transform(y_test_tens.detach().numpy())

    # Graph training loss, Actual Stock Price, Predicted Stock Price
    visualize_results(stock, reference_frame, train_loss, model_train, model_test, y_train, y_test)

    while True:
        try:
            re_run = input("Would you like to start over? (Y or N): ")
            if re_run == 'y'.casefold():
                # Close connections, restart program
                try:
                    p = psutil.Process(os.getpid())
                    for handler in p.connections():
                        os.close(handler.fd)
                except Exception as e:
                    print(e)

                python = sys.executable
                os.execl(python, python, *sys.argv)
            elif re_run == 'n'.casefold():
                sys.exit(0)
        except ValueError:
            print("Enter Y to reset, N to exit.")
