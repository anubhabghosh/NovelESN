"""
This program writes down the main for recurrent networks (like ESN, AR-Model, RNN)
after restructuring the code appropriately so that different models can be easily 
compared

NOTE: NovelESN Code was originally written by Aleix Espuña Fontcuberta, and that part
of the code is simply restructured here, without changing original functionalities 
much

Author: Anubhab Ghosh
"""

# Import the necessary libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_msah_training_dataset, get_minimum, concat_data, get_cycle
from src.utils import normalize, create_list_of_dicts, save_pred_results, unnormalize, plot_losses
import torch
import copy
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import json
import itertools
import argparse
from novel_esn import NovelEsn
from armodel import Linear_AR, create_dataset, train_armodel, predict_armodel
from argparse import RawTextHelpFormatter
from cycle_selector_dynamo import get_train_test_dynamo
from cycle_selector_realsolar import get_train_test_realsolar, plot_train_test_data
from testrnn_aliter import RNN_model, train_rnn, predict_rnn, train_validation_split
from train_novelESN import train_model_ESN
from train_linear_ar import train_model_AR
from train_recurrent_nets import train_model_RNN

def get_pred_output_file(filename, model_name):

    filename_, extension = filename.split(".")
    filename_ = filename_.lower()
    if (not model_name is None) and (model_name in filename_):
        filename_new = filename_ + "." + extension
    elif not model_name is None:
        filename_new = filename_ + "_" + model_name + "." + extension
    return filename_new

def main():
    
    parser = argparse.ArgumentParser(description=
    "Use a variety of recurrent architectures for predicting solar sunpots as a time series\n"\
    "Example: python main_restructured.py --model_type [esn/linear_ar/rnn/lstm/gru] --dataset dynamo --train_file [full path to training data file] \
    --output_file [path to file containing predictions] --predict_cycle_num [index of cycle to be predicted] --grid_search [0/1] --compare_all [0/1] \n"
    "Description of different model types: \n"\
    "esn: echo state network,\n" \
    "linear_ar: linear autoregressive model, \n"\
    "rnn: simple recurrent network (vanilla RNN), \n" \
    "lstm: long-short term memory network, \n" \
    "gru: gated recurrent units (simplification of lstm architecture)", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--model_type", help="Enter the desired model", default="esn", type=str)
    parser.add_argument("--dataset", help="Type of dataset used - (dynamo/solar_data/sinus)", default="dynamo", type=str)
    parser.add_argument("--train_file", help="Location of training data file", default=None, type=str)
    parser.add_argument("--output_file", help="Location of the output file", default=None, type=str)
    parser.add_argument("--predict_cycle_num", help="Cycle index to be predicted", default=None, type=int)
    parser.add_argument("--grid_search", help="Option to perform grid search or not (1 - True, 0 - False)", default=0, type=int)
    parser.add_argument("--compare_all", help="Option to compare all models or not (1 - True, 0 - False)", default=0, type=int)

    # Parse the arguments
    args = parser.parse_args() 
    model_type = args.model_type.lower()
    dataset = args.dataset
    train_file = args.train_file
    output_file = args.output_file
    use_grid_search = args.grid_search
    compare_all_models = args.compare_all
    predict_cycle_num = args.predict_cycle_num

    # Load the configurations required for training
    # It is assumed that the configurations are present in this location
    config_file = "./configurations_{}.json".format(dataset)  
    with open(config_file) as f:
        options = json.load(f)  # This loads options as a dict with keys that can be accessed
    
    # Load the training data
    data = np.loadtxt(train_file)

    # Keep a copy of the unnormalized data
    unnormalized_data = copy.deepcopy(data)
    data[:, 1], Xmax, Xmin = normalize(X=data[:, 1], feature_space=(0, 1))
    minimum_idx = get_minimum(data, dataset)

    # In case running for a single model
    #TODO: Ensure that 'tr_verbose' is a calling parameter and usage is according to 'compare_all_models' flag
    #      Right now, the plot commands are commented out
    #TODO: Fix save model feature in case of comparison of all models
    if compare_all_models == 0:

        output_file = get_pred_output_file(output_file, model_type)
        if model_type == "esn":
            predictions_esn = train_model_ESN(options, model_type, data, minimum_idx, predict_cycle_num=predict_cycle_num, 
                                            tau=1, output_file=output_file)

        elif model_type == "linear_ar":
            predictions_ar = train_model_AR(options, model_type, data, minimum_idx, predict_cycle_num=predict_cycle_num, 
                                            tau=1, output_file=output_file, use_grid_search=use_grid_search)

        elif model_type in ["rnn", "lstm", "gru"]:
            predictions_rnn = train_model_RNN(options, model_type, data, minimum_idx, predict_cycle_num=predict_cycle_num, 
                                            tau=1, output_file=output_file, use_grid_search=use_grid_search, Xmax=Xmax, Xmin=Xmin)

    # In case running for all models
    elif compare_all_models == 1:

        predictions_esn, ytest = train_model_ESN(options, "esn", data, minimum_idx, predict_cycle_num=predict_cycle_num, 
                                        tau=1, output_file=get_pred_output_file(output_file, "esn"))
        predictions_ar = train_model_AR(options, "linear_ar", data, minimum_idx, predict_cycle_num=predict_cycle_num, tau=1, 
                                        output_file=get_pred_output_file(output_file, "linear_ar"), use_grid_search=use_grid_search)
        #predictions_vanilla_rnn = train_model_RNN(options, "rnn", data, minimum_idx, predict_cycle_num=predict_cycle_num, tau=1, 
        #                                        output_file=output_file, use_grid_search=use_grid_search)
        predictions_lstm = train_model_RNN(options, "lstm", data, minimum_idx, predict_cycle_num=predict_cycle_num, tau=1, 
                                        output_file=get_pred_output_file(output_file, "lstm"), use_grid_search=use_grid_search,
                                        Xmax=Xmax, Xmin=Xmin)
        predictions_gru = train_model_RNN(options, "gru", data, minimum_idx, predict_cycle_num=predict_cycle_num, tau=1, 
                                        output_file=get_pred_output_file(output_file, "gru"), use_grid_search=use_grid_search,
                                        Xmax=Xmax, Xmin=Xmin)

        # Plot the LSTM, GRU predictions
        #plt.figure(figsize=(15,10))
        plt.figure()
        plt.title("Compare predictions across models", fontsize=20)
        if len(ytest) > 0:

            plt.plot(ytest[:,0], ytest[:,1], '+-', label="actual test signal", color="orange")
            plt.plot(ytest[:,0], predictions_esn, 'o-', label="ESN prediction", color="red")
            plt.plot(ytest[:,0], predictions_ar, '+-', label="AR prediction", color="cyan")
            #plt.plot(ytest[:,0], predictions_vanilla_rnn, '.-', label="RNN prediction", color="pink")
            plt.plot(ytest[:,0], predictions_lstm, 'x-', label="LSTM prediction", color="blue")
            plt.plot(ytest[:,0], predictions_gru, '*-', label="GRU prediction", color="green")
            plt.legend(fontsize=16)

        else:

            plt.plot(predictions_esn, 'o-', label="ESN prediction", color="red")
            plt.plot(predictions_ar, '+-', label="AR prediction", color="cyan")
        #    plt.plot(predictions_vanilla_rnn, '.-', label="RNN prediction", color="pink")
            plt.plot(predictions_lstm, 'x-', label="LSTM prediction", color="blue")
            plt.plot(predictions_gru, '*-', label="GRU prediction", color="green")
            plt.legend(fontsize=16)

        plt.savefig('./log/ComparingPred_Cycle{}.pdf'.format(predict_cycle_num))
        plt.show()

    
if __name__ == "__main__":
    main()

