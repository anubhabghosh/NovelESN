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
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import json
import argparse
from novel_esn import NovelEsn
from armodel import Linear_AR, create_dataset, train_armodel, predict_armodel, plot_losses
from argparse import RawTextHelpFormatter
from cycle_selector_dynamo import get_train_test_dynamo
from cycle_selector_realsolar import get_train_test_realsolar, plot_train_test_data

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
    #TODO: AR model integrated
    elif model_type.lower() == "linear_ar":

        model = Linear_AR(
                        num_taps=options[model_type]["num_taps"],
                        lossfn_type=options[model_type]["lossfn_type"],
                        lr=options[model_type]["lr"],
                        num_epochs=options[model_type]["num_epochs"],
                        init_net=options[model_type]["init_net"],
                        device=options[model_type]["device"] 
        )

    return model

def train_and_predict_ESN(model, train_data, test_data=None):
   
    model.teacher_forcing(train_data)
    train_mse = model.train(train_signal=train_data)
    print("training mse over q =", train_mse / model.yntau_size_q)
    predictions, pred_indexes = model.predict()
    test_mse = mean_squared_error(test_data, predictions)
    print("test mse=", test_mse)
    
    plot_predictions(actual_test_data=test_data,
                    pred_indexes=pred_indexes,
                    predictions=predictions,
                    title="Predictions using NovelESN model")
        
    return predictions, pred_indexes

def train_and_predict_AR(model, train_data, test_data=None, 
                    train_indices=None, test_indices=None):

    train_data_inputs, train_data_targets = create_dataset(train_data, alen=model.num_taps)
    #test_data_inputs, test_data_targets = create_dataset(test_data, alen=model.num_taps)
    tr_losses, val_losses, model = train_armodel(model, nepochs=model.num_epochs, inputs=train_data_inputs,
        targets=train_data_targets, tr_split=0.75)

    #test_input = test_data_inputs[0].reshape((1, -1))
    test_input = train_data[-model.num_taps:].reshape((1, -1))
    predictions_ar = predict_armodel(model=model, eval_input=test_input, n_predict=len(test_data))
    
    plot_predictions(
        actual_test_data=test_data,
        pred_indexes=test_indices,
        predictions=predictions_ar,
        title="Predictions using Linear AR model"
    )

    return predictions_ar, test_indices

def plot_predictions(actual_test_data, pred_indexes, predictions, title):

    #Prediction plot
    plt.figure()
    #plt.title("Prediction value of number of sunspots vs time index", fontsize=20)
    plt.title(title, fontsize=20)
    plt.plot(pred_indexes, actual_test_data, label="actual test signal", color="orange")
    plt.plot(pred_indexes, predictions, label="prediction", color="green")
    plt.legend()
    plt.show()

def main():
    
    parser = argparse.ArgumentParser(description=
    "Use a variety of recurrent architectures for predicting solar sunpots as a time series\n"\
    "Example: python main.py --model_type [esn/linear_ar/rnn/lstm/gru] --dataset dynamo --train_file [full path to training data file] \
    --output_file [path to file containing predictions] --test_file [path to test file (if any)] \
    --verbosity [1 or 2] \n"
    "Description of different model types: \n"\
    "esn: echo state network,\n" \
    "linear_ar: linear autoregressive model, \n"\
    "rnn: simple recurrent network (vanilla RNN / Elman unit), \n" \
    "lstm: long-short term memory network, \n" \
    "gru: gated recurrent units (simplification of lstm architecture)", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model_type", help="Enter the desired model", default="esn", type=str)
    parser.add_argument("--dataset", help="Type of dataset used - (dynamo/solar_data/sinus)", default="dynamo", type=str)
    parser.add_argument("--train_file", help="Location of training data file", default=None, type=str)
    parser.add_argument("--output_file", help="Location of the output file", default=None, type=str)
    #parser.add_argument("--test_file", help="(Optional) Location of the test data file", default=None, type=str)
    parser.add_argument("--use_data", help="(Optional) Use Preprocessed data, 1 - use already present data, 0 - use custom split", default=0, type=int)
    parser.add_argument("--predict_cycle_num", help="Cycle number to be predicted", default=None, type=int)

    # Parse the arguments
    args = parser.parse_args() 
    model_type = args.model_type
    dataset = args.dataset
    train_file = args.train_file
    output_file = args.output_file
    #test_file = args.test_file
    use_data_flag = args.use_data
    predict_cycle_num = args.predict_cycle_num

    # Load the configurations required for training
    config_file = "./configurations.json" # It is assumed that the configurations are 
                                          # present in this location

    with open(config_file) as f:
        options = json.load(f) # This loads options as a dict with keys that can be accessed

    print(use_data_flag)
    # Load the training data
    if use_data_flag == 1:
        
        train_data = np.loadtxt("./data/TrainingSignals/dynamo_train.txt") # Only contains the signal
        test_data = np.loadtxt("./data/TestSignals/dynamo_test_tau151_q152.txt") # Only contains the signal
        
        #Training data plot
        plt.figure()
        plt.title("Training data vs time index")
        plt.plot(train_data)
        plt.show()

        #train_data = np.loadtxt(train_file)
        # Load the test data (if any)
        #if test_file:
        #    test_data = np.loadtxt(test_file)
        # Load the model with corresponding options
        model = load_model_with_opts(options, model_type)
        #pred of q values
        predictions, pred_indexes = train_and_predict_ESN(model, train_data, test_data)

        #given a training time series y_0, y_1, y_2 ...y_M-1, the program will predict:
        # y_M+tau-(q-1), y_M+tau-(q-2) ... y_M+tau
        np.savetxt(fname = output_file, X = predictions)
    
    else:

        train_test_data = np.loadtxt(train_file)
        if dataset.lower() == "dynamo":
            tr_data_time, tr_data_signal, te_data_time, te_data_signal = get_train_test_dynamo(time=train_test_data[:,0],
                                                                                            dynamo=train_test_data[:,1],
                                                                                            cycle_num=predict_cycle_num)
        elif dataset.lower() == "solar":
            tr_data_time, tr_data_signal, te_data_time, te_data_signal = get_train_test_realsolar(time=train_test_data[:,0],
                                                                                            dynamo=train_test_data[:,1],
                                                                                            cycle_num=predict_cycle_num)
        plot_train_test_data(trdata_time=tr_data_time,
                            trdata_signal=tr_data_signal,
                            tedata_time=te_data_time,
                            tedata_signal=te_data_signal)
        
        #NOTE: This modification is only applicable for NovelESN
        if model_type == "esn":
            options["esn"]["tau"] = len(te_data_signal) - 1
            options["esn"]["history_q"] = options["esn"]["tau"] + 1
            print("Shape of training data:{}".format(tr_data_signal.shape))
            print("Shape of testing data:{}".format(te_data_signal.shape))
            # Load the model with corresponding options
            model = load_model_with_opts(options, model_type)
            #pred of q values
            predictions, pred_indexes = train_and_predict_ESN(model, tr_data_signal, te_data_signal)

        elif model_type == "linear_ar":

            # Load the model with corresponding options
            model = load_model_with_opts(options, model_type)
            #pred of q values
            predictions, pred_indexes = train_and_predict_AR(model, tr_data_signal, te_data_signal, 
                tr_data_time, te_data_time)

        # Save the results in the output file
        np.savetxt(fname = output_file, X = predictions)

if __name__ == "__main__":
    main()