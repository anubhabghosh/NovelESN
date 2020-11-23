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
from rnntest import RNN, train_and_predict_RNN

def load_model_with_opts(options, model_type):
    """ This function is used for loading the appropriate model
    with the configuration details mentioned in the 'options' (.json)
    file

    Args:
        options ([json_dict]): dictionary containing the list of options 
        for each model considered
        model_type ([str]): type of model under consideration
    """

    if model_type == "esn":

        model = NovelEsn(
                        num_neur=options[model_type]["num_neurons"],
                        conn_per_neur=options[model_type]["conn_per_neuron"],
                        spectr_rad=options[model_type]["spectral_radius"],
                        tau=options[model_type]["tau"],
                        history_q=options[model_type]["history_q"],
                        history_p=options[model_type]["history_p"],
                        beta_regularizer=options[model_type]["beta_regularizer"]
                        )
    # TODO: AR model integrated
    elif model_type == "linear_ar":

        model = Linear_AR(
                        num_taps=options[model_type]["num_taps"],
                        lossfn_type=options[model_type]["lossfn_type"],
                        lr=options[model_type]["lr"],
                        num_epochs=options[model_type]["num_epochs"],
                        init_net=options[model_type]["init_net"],
                        device=options[model_type]["device"] 
        )
    elif model_type == "rnn":
        model = RNN()
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


def train_and_predict_AR(model, train_data_inputs, train_data_targets, test_data):
    tr_losses, val_losses, model = train_armodel(model, nepochs=model.num_epochs, inputs=train_data_inputs,
        targets=train_data_targets, tr_split=0.75)

    predictions_ar = predict_armodel(model=model, eval_input=train_data_inputs[-1], n_predict=len(test_data))

    return predictions_ar

def plot_predictions(ytest,  predictions, title):

    #Prediction plot
    plt.figure()
    #plt.title("Prediction value of number of sunspots vs time index", fontsize=20)
    plt.title(title, fontsize=20)
    plt.plot(ytest[:,0], ytest[:,1], label="actual test signal", color="orange")
    plt.plot(ytest[:,0], predictions, label="prediction", color="green")
    plt.legend()
    plt.show()
"""

    plot_predictions(
        actual_test_data=test_data,
        pred_indexes=test_indices,
        predictions=predictions_ar,
        title="Predictions using Linear AR model"
    )
"""
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
    parser.add_argument("--verbose", help="Verbosity (0 or 1)", default=0, type=int)
    #parser.add_argument("--test_file", help="(Optional) Location of the test data file", default=None, type=str)
    parser.add_argument("--predict_cycle_num", help="Cycle number to be predicted", default=None, type=int)

    # Parse the arguments
    args = parser.parse_args() 
    model_type = args.model_type.lower()
    dataset = args.dataset
    train_file = args.train_file
    output_file = args.output_file
    verbose = args.verbose

    # test_file = args.test_file
    predict_cycle_num = args.predict_cycle_num

    # Load the configurations required for training
    config_file = "./configurations.json"  # It is assumed that the configurations are
                                           # present in this location

    with open(config_file) as f:
        options = json.load(f)  # This loads options as a dict with keys that can be accessed
    options[model_type]["num_taps"] = 10
    p = options[model_type]["num_taps"]

    # Load the training data
    data = np.loadtxt(train_file)
    data[:, 1] = 2*((data[:, 1] - data[:,1].min())/(data[:,1].max() - data[:,1].min())) - 1
    minimum_idx = get_minimum(data, dataset)
    plt.figure()


    # Get multiple step ahead prediction datasets

    X, Y = get_msah_training_dataset(data, minimum_idx, tau=1, p=p)
    # options[model_type]["num_taps"]
    n_cycles = len(Y)
    n_tests = 3

    # xtrain, ytrain, ytest = get_cycle(X, Y, n_cycles+1)
    P = [10, 20, 30]
    val_err = np.zeros((n_cycles, len(P)))

        # errors = new_train_ar(data,minimum_idx)
        # errors = {"validatation errors": (n_val_cycles, n_tried_numtapsvalues),
        #            "test_errors":(n_test_cycles,),
        #            "test_predictions: list of n_test_cycles arrays [ (length of 1st test cycle, 2), .. ]
        #            "future_points": (120,)
        #  }

    for ip, p in enumerate(P):
        X, Y = get_msah_training_dataset(data, minimum_idx, tau=1, p=p)
        for icycle in range(n_cycles-n_tests):
            xtrain, ytrain, yval = get_cycle(X, Y, icycle)
            if model_type == "linear_ar":
                model = Linear_AR(
                    num_taps=p,
                    lossfn_type=options[model_type]["lossfn_type"],
                    lr=options[model_type]["lr"],
                    num_epochs=options[model_type]["num_epochs"],
                    init_net=options[model_type]["init_net"],
                    device=options[model_type]["device"]
                )

                predictions = train_and_predict_AR(model, concat_data(xtrain), concat_data(ytrain), yval[:, 1])

            elif model_type == "rnn":
                #Usage:
                #  python /home/anthon@ad.cmm.se/Desktop/projects/NovelESN/main.py --model_type rnn --dataset dynamo --train_file data/dynamo_esn.txt --output_file tmp.txt --predict_cycle_num 10
                X, Y = get_msah_training_dataset(data, minimum_idx, tau=1, p=np.inf)

                predictions = train_and_predict_RNN(X, Y, enplot=False,n_future=120)
                sys.exit(0)
            val_err[icycle, ip] = mean_squared_error(yval[:, 1], predictions)


    optimal_p = np.argmin(val_err.mean(0)).reshape(-1)[0]
    X, Y = get_msah_training_dataset(data, minimum_idx, tau=1, p=optimal_p)
    test_err_ar=np.zeros(n_tests)
    for i_test_cycle in range(n_cycles-n_tests, n_cycles):
        xtrain, ytrain, ytest = get_cycle(X, Y, i_test_cycle)
        model = load_model_with_opts(options, model_type)
        predictions = train_and_predict_AR(model, concat_data(xtrain), concat_data(ytrain), yval[:, 1])
        test_err_ar[i_test_cycle] = mean_squared_error(ytest[:, 1], predictions)

    # model = load_model_with_opts(options, model_type)
    model = RNN(input_size=p, hidden_size=10)
    predictions = train_and_predict_RNN(model, concat_data(xtrain), concat_data(ytrain), ytest[:, 1])

    err[icycle] = mean_squared_error(ytest[:, 1], predictions)

    plot_predictions(
       ytest=ytest,
       predictions=predictions,
       title="Predictions using Linear AR model"
    )

    plt.figure();
    plt.plot(list(range(n_cycles)), err)
    plt.show()
    sys.exit(0)



    if model_type == "esn":
        options["esn"]["tau"] = len(te_data_signal) - 1
        options["esn"]["history_q"] = options["esn"]["tau"] + 1
        print("Shape of training data:{}".format(tr_data_signal.shape))
        print("Shape of testing data:{}".format(te_data_signal.shape))
        # Load the model with corresponding options
        model = load_model_with_opts(options, model_type)
        # pred of q values
        predictions, pred_indexes = train_and_predict_ESN(model, tr_data_signal, te_data_signal)

    elif model_type == "linear_ar":
        # Load the model with corresponding options
        model = load_model_with_opts(options, model_type)

        # pred of q values
        predictions, pred_indexes = train_and_predict_AR(model, xtrain, ytrain, ytest)

    elif model_type == "rnn":
        model = load_model_with_opts(options, model_type)



    with open("results__{}.txt".format(model_type), "a") as fp:
        print("\t".join(
                    ["{}:{}".format(k, v) for k, v in options["linear_ar"].items()]
                    + ["{}:{}".format("test__mse", ((predictions-te_data_signal)**2).mean())]
                    + ["{}:{}".format("train__mse", ((predictions - te_data_signal) ** 2).mean())]
                    + ["{}:{}".format("val__mse", ((predictions - te_data_signal) ** 2).mean())]
                    ), file=fp)

    # Save the results in the output file
    np.savetxt(fname=output_file,
               X=np.concatenate([predictions.reshape(-1, 1), te_data_signal.reshape(-1, 1)], axis=1)
               )


if __name__ == "__main__":
    main()

