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
from src.utils import get_msah_training_dataset, get_minimum, concat_data, get_cycle, normalize, create_list_of_dicts
import torch
import copy
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import json
import itertools
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


def train_and_predict_AR(model, train_data_inputs, train_data_targets, test_data, tr_to_val_split=0.9, tr_verbose=False):
    
    # Apply concat data to concatenate the rows that have columns with signal (not the timestamp)
    train_data_inputs, train_data_targets = concat_data(train_data_inputs), concat_data(train_data_targets) 

    tr_losses, val_losses, model = train_armodel(model, nepochs=model.num_epochs, inputs=train_data_inputs,
        targets=train_data_targets, tr_split=tr_to_val_split, tr_verbose=tr_verbose)
    
    if len(test_data) > 0:
        predictions_ar = predict_armodel(model=model, eval_input=train_data_inputs[-1], n_predict=len(test_data))
        test_error = mean_squared_error(y_true=test_data[:, -1], y_pred=predictions_ar)
    else:
        #NOTE: Heuristically setting the number of future predictions
        predictions_ar = predict_armodel(model=model, eval_input=train_data_inputs[-1], n_predict=132)
        test_error = np.nan
    
    tr_error = tr_losses[-1] # latest training error
    val_error = val_losses[-1] # latest validation error
    #print("**********************************************************************************************************")
    print("{} - {},  {} - {},  {} - {:.8f},  {} - {:.8f},  {}, - {:.8f}".format(
                                                                "Model", "AR",
                                                                "P",
                                                                model.num_taps,
                                                                "Training Error",
                                                                tr_error,
                                                                "Validation Error",
                                                                val_error,
                                                                "Test Error",
                                                                test_error))
    print("***********************************************************************************************************")
    '''
    with open("results__{}.txt".format(model_type), "a") as fp:
        print("**********************************************************************************************************")
        print("{} - {},  {} - {},  {} - {:.8f},  {} - {:.8f},  {}, - {:.8f}".format(
                                                                "Model", "AR",
                                                                "P",
                                                                model.num_taps,
                                                                "Training Error",
                                                                tr_error,
                                                                "Validation Error",
                                                                val_error,
                                                                "Test Error",
                                                                test_error), fp)
        print("***********************************************************************************************************")
    '''
    return predictions_ar, test_error, val_error, tr_error

def plot_predictions(ytest,  predictions, title):

    #Prediction plot
    plt.figure()
    #plt.title("Prediction value of number of sunspots vs time index", fontsize=20)
    plt.title(title, fontsize=10)
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

def grid_search_AR_all_cycles(data, solar_indices, model_type, options, params, 
                predict_cycle_num_array):
    
    assert len(list(params.keys())) == 1 #TODO: For now, this works for only num_taps and 1 parameter
    assert list(params.keys()) == ["num_taps"]
    
    # Matrices to store validation, training errors for each cycle asked to predict and 
    # for each value of "num_taps"

    if type(predict_cycle_num_array) != list:
        predict_cycle_num_array = list(predict_cycle_num_array)

    val_errors_all = np.zeros((len(predict_cycle_num_array), len(params["num_taps"])))
    training_errors_all = np.zeros((len(predict_cycle_num_array), len(params["num_taps"])))
    test_errors_all = np.zeros((len(predict_cycle_num_array), len(params["num_taps"])))

    test_predictions = []
    num_future_points = np.mean(np.diff(solar_indices)) # Mean of cycle length
    #future_points = np.zeros((num_future_points, 1))

    for i, pr_cycle_num in enumerate(predict_cycle_num_array):

        training_errors_i, val_errors_i, test_errors_i = grid_search_AR_single_cycle(data, solar_indices,
            model_type, options, params, pr_cycle_num)

        training_errors_all[i,:] = training_errors_i
        val_errors_all[i,:] = val_errors_i
        test_errors_all[i,:] = test_errors_i
    
    optimal_num_taps_all_idx = np.argmin(val_errors_all, axis=1)
    optimal_num_taps_all = params["num_taps"][int(optimal_num_taps_all_idx)]
    
    return optimal_num_taps_all, training_errors_all, val_errors_all, test_errors_all

def grid_search_AR_single_cycle(data, solar_indices, model_type, options, params, 
                predict_cycle_num):

    params_dict_list_all = create_list_of_dicts(options=options,
                                                model_type=model_type,
                                                param_dict=params)
    val_errors = np.zeros((1, len(params["num_taps"])))
    training_errors = np.zeros((1, len(params["num_taps"])))
    test_errors = np.zeros((1, len(params["num_taps"])))
    #prediction_array = []

    for i, param_options in enumerate(params_dict_list_all):

        p = param_options["num_taps"]
        assert (p > 0) == True, print("Invalid order specified as parameter")
        print("Parameter set used:\n{}".format(param_options))
        
        X, Y = get_msah_training_dataset(data, minimum_idx=solar_indices,tau=1, p=p)
        xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num - 1)
        options[model_type]["num_taps"] = p
        model = load_model_with_opts(options, model_type)
        predictions, test_error, val_error, tr_error = train_and_predict_AR(
            model, xtrain, ytrain, ytest)

        val_errors[:, i] = val_error
        training_errors[:, i] = tr_error
        test_errors[:, i] = test_error

    return training_errors, val_errors, test_errors

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
    parser.add_argument("--grid_search", help="Option to perform grid search or not (1 - True, 0 - False", default=0, type=int)

    # Parse the arguments
    args = parser.parse_args() 
    model_type = args.model_type.lower()
    dataset = args.dataset
    train_file = args.train_file
    output_file = args.output_file
    verbose = args.verbose
    use_grid_search = args.grid_search

    # test_file = args.test_file
    predict_cycle_num = args.predict_cycle_num

    # Load the configurations required for training
    config_file = "./configurations.json"  # It is assumed that the configurations are
                                           # present in this location

    with open(config_file) as f:
        options = json.load(f)  # This loads options as a dict with keys that can be accessed
    
    # Load the training data
    data = np.loadtxt(train_file)
    # Keep a copy of the unnormalized data
    unnormalized_data = copy.deepcopy(data)
    data[:, 1] = normalize(X=data[:, 1], feature_space=(0, 1))
    minimum_idx = get_minimum(data, dataset)

    # Get multiple step ahead prediction datasets : #NOTE: Only for Linear_AR so far
    if model_type == "esn":

        #if dataset == "dynamo":
        #    tr_data_time, tr_data_signal, te_data_time, te_data_signal = get_train_test_dynamo(data[:,0],
                                                                                               #data[:,1],
                                                                                               #cycle_num=predict_cycle_num)
        #elif dataset == "solar":
        #    tr_data_time, tr_data_signal, te_data_time, te_data_signal = get_train_test_realsolar(data[:,0],
                                                                                               #data[:,1],
                                                                                               #cycle_num=predict_cycle_num)
        #options["esn"]["tau"] = len(te_data_signal) - 1
        #options["esn"]["history_q"] = options["esn"]["tau"] + 1
        #print("Shape of training data:{}".format(tr_data_signal.shape))
        #print("Shape of testing data:{}".format(te_data_signal.shape))
        # Load the model with corresponding options
        #model = load_model_with_opts(options, model_type)
        # pred of q values#
        # predictions, pred_indexes = train_and_predict_ESN(model, tr_data_signal, te_data_signal)
        pass

    elif model_type == "linear_ar":
        # Load the model with corresponding options
        if use_grid_search == 0:
            
            model = load_model_with_opts(options, model_type)
            X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx,tau=1, p=options[model_type]["num_taps"])
            # predict cycle index = entered predict cycle num - 1
            xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num - 1)
            # pred of q values
            predictions_ar, test_error, val_error, tr_error = train_and_predict_AR(model, xtrain, ytrain, ytest, tr_to_val_split=0.9, tr_verbose=True)
            plot_predictions(predictions=predictions_ar, ytest=ytest, title="AR model predictions with {} taps for cycle index {}".format(
                options[model_type]["num_taps"], predict_cycle_num))

        elif use_grid_search == 1:
            
            Error_dict = {}
            test_predictions = []
            test_error_optimal = []
            nval = 1
            num_total_cycles = len(np.diff(minimum_idx))
            #predict_cycle_num_array = list(np.arange(num_total_cycles-nval, num_total_cycles))
            predict_cycle_num_array = [predict_cycle_num]
            #params = {"num_taps":[5,10,15]}
            params = {"num_taps":list(np.arange(10, 40, 2))}
            #TODO: Fix array nature of optimal_num_taps_all
            optimal_num_taps_all, training_errors_all, val_errors_all, test_errors_all = grid_search_AR_all_cycles(data=data,
                solar_indices=minimum_idx, model_type=model_type, options=options, params=params, predict_cycle_num_array=predict_cycle_num_array)
            
            Error_dict["validation_errors"] = val_errors_all.tolist()
            Error_dict["Optimal_num_taps"] = optimal_num_taps_all.tolist()

            plt.figure()
            plt.plot(params["num_taps"], val_errors_all[0], label="Validation MSE")
            plt.plot(params["num_taps"], training_errors_all[0], label="Training MSE")
            plt.ylabel("MSE")
            plt.xlabel("Number of taps")
            plt.legend()
            plt.title("Error (MSE) vs number of taps")
            plt.show()

            if type(optimal_num_taps_all) != list:
                optimal_num_taps_all = [optimal_num_taps_all]

            # Retrain the model again with the optimal value
            for i, optimal_num_taps in enumerate(optimal_num_taps_all):
                
                options[model_type]["num_taps"] = optimal_num_taps
                model = load_model_with_opts(options, model_type)
                X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx,tau=1, p=optimal_num_taps)
                xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num_array[i] - 1)
                # pred of q values
                predictions_ar, test_error, val_error, tr_error = train_and_predict_AR(model, xtrain, ytrain, ytest, tr_to_val_split=0.9, tr_verbose=True)
                test_predictions.append(predictions_ar)
                if len(ytest) > 0:
                    
                    plot_predictions(predictions=predictions_ar, ytest=ytest, title="AR model predictions with {} taps for cycle index {}".format(
                        optimal_num_taps, predict_cycle_num_array[i]))
                    test_error_optimal.append(test_error)
                
                else:
                    
                    resolution = np.around(np.diff(data[:,0]).mean(),1)
                    plt.figure()
                    plt.plot(data[:minimum_idx[-1],0], data[:minimum_idx[-1],1], 'r+-')
                    plt.plot(np.arange(ytrain[-1][-1][0] + resolution, ((len(predictions_ar)) * resolution) + ytrain[-1][-1][0], resolution), predictions_ar, 'b*-')
                    plt.legend(['Original timeseries', 'Future prediction'])
                    plt.title('Plot of original timeseries and future predictions')
                    plt.show()

            Error_dict["Test_predictions"] = test_predictions
            Error_dict["Test_error"] = [test_error_optimal]

            #with open('./log/grid_search_results.json', 'w') as fp:
            #    json.dump(Error_dict, fp, sort_keys=False, indent=4)

            #TODO: To fix saving result files properly

    elif model_type == "rnn":
        model = load_model_with_opts(options, model_type)
    '''
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
    '''

if __name__ == "__main__":
    main()

