"""
This program writes down the main for recurrent networks (like ESN, AR-Model, RNN)
after restructuring the code appropriately so that different models can be easily 
compared

NOTE: NovelESN Code was originally written by Aleix Espuña Fontcuberta, and that part
of the code is simply restructured here, without changing original functionalities much

Author: Anubhab Ghosh
"""

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import json
import argparse
from novel_esn import NovelEsn
from argparse import RawTextHelpFormatter

def load_model_with_opts(options, model_type):
    """ This function is used for loading the appropriate model
    with the configuration details mentioned in the 'options' (.json)
    file

    Args:
        options ([json_dict]): dictionary containing the list of options 
        for each model considered
        model_type ([str]): type of model under consideration
    """
    if model_type.lower() == "esn":

        model = NovelEsn(
                        num_neur=options[model_type]["num_neurons"],
                        conn_per_neur=options[model_type]["conn_per_neuron"],
                        spectr_rad=options[model_type]["spectral_radius"],
                        tau=options[model_type]["tau"],
                        history_q=options[model_type]["history_q"],
                        history_p=options[model_type]["history_p"],
                        beta_regularizer=options[model_type]["beta_regularizer"]
                        )

    return model

def train(model, model_type, options, train_data, test_data=None):

    if model_type.lower() == "esn":
        
        model.teacher_forcing(train_data)
        prediction, train_mse, pred_indexes = model.predict(train_data)
        print("training mse over q =", train_mse / model.yntau_size_q)
        
        #Training data plot
        plt.figure()
        plt.title("Training data vs time index")
        plt.plot(train_data)
        plt.show()

        #Prediction plot
        plt.figure()
        plt.title("Prediction vs time index")
        plt.plot(pred_indexes, prediction, label="prediction", color="green")
        if test_data.all():
            plt.plot(pred_indexes, test_data, label="test signal", color="orange")
            test_mse = mean_squared_error(test_data, prediction)
            print("test mse=", test_mse)
        plt.legend()
        plt.show()
        
        return prediction, train_mse
    

def main():
    
    parser = argparse.ArgumentParser(description=
    "Use a variety of recurrent architectures for predicting solar sunpots as a time series\n"\
    "Example: python main.py --model_type [esn/linear_ar/rnn/lstm/gru] --train_file [full path to training data file] \
    --output_file [path to file containing predictions] --test_file [path to test file (if any)] \
    --verbosity [1 or 2] \n"
    "Description of different model types: \n"\
    "esn: echo state network,\n" \
    "linear_ar: linear autoregressive model, \n"\
    "rnn: simple recurrent network (vanilla RNN / Elman unit), \n" \
    "lstm: long-short term memory network, \n" \
    "gru: gated recurrent units (simplification of lstm architecture)", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model_type", help="Enter the desired model", default="esn", type=str)
    parser.add_argument("--train_file", help="Location of training data file", default=None, type=str)
    parser.add_argument("--output_file", help="Location of the output file", default=None, type=str)
    parser.add_argument("--test_file", help="(Optional) Location of the test data file", default=None, type=str)
    parser.add_argument("--verbosity", help="(Optional) Increase output verbosity, 1 - only output, 2 - output + input", type=int)
    
    # Parse the arguments
    args = parser.parse_args() 
    model_type = args.model_type
    train_file = args.train_file
    output_file = args.output_file
    test_file = args.test_file
    verbosity = args.verbosity

    # Load the configurations required for training
    config_file = "./RecurrentNNsforSunspots/configurations.json" # It is assumed that the configurations are 
                                          # present in this location

    with open(config_file) as f:
        options = json.load(f) # This loads options as a dict with keys that can be accessed

    # Load the training data
    train_data = np.loadtxt(train_file)
    
    # Load the test data (if any)
    if test_file:
        test_data = np.loadtxt(test_file)

    # Load the model with corresponding options
    model = load_model_with_opts(options, model_type)
    #pred of q values
    prediction, train_mse = train(model, model_type, options, train_data, test_data)

    #given a training time series y_0, y_1, y_2 ...y_M-1, the program will predict:
    # y_M+tau-(q-1), y_M+tau-(q-2) ... y_M+tau
    np.savetxt(fname = output_file, X = prediction)

    #index_last_pred = len(train_signal) + tau # = M+tau
    #index_first_pred = index_last_pred-(yntau_size_q-1) # = M+tau-(q-1)
    #pred_indexes = np.arange(index_first_pred, index_last_pred+1)

if __name__ == "__main__":
    main()