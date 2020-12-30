# Import the necessary libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_msah_training_dataset, get_minimum, concat_data, get_cycle, plot_predictions, plot_future_predictions
from src.utils import normalize, create_list_of_dicts, save_pred_results, unnormalize, plot_losses, count_params
import torch
import copy
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import json
import itertools
import argparse
from testrnn_aliter import train_rnn, predict_rnn, train_validation_split, RNN_model
from load_model import load_model_with_opts


# Implement the version here
def train_and_predict_RNN(model, train_data_inputs, train_data_targets, test_data, tr_to_val_split=0.9, tr_verbose=False):

    # Count number of model parameters
    total_num_params, total_num_trainable_params = count_params(model=model)
    print("The total number of params: {} and the number of trainable params:{}".format(total_num_params, total_num_trainable_params))

    # Apply concat data to concatenate the rows that have columns with signal (not the timestamp)
    train_data_inputs, train_data_targets = concat_data(train_data_inputs), concat_data(train_data_targets) 
    
    if len(train_data_inputs.shape) == 2:
        # Extra dimension to be added
        N, P = train_data_inputs.shape
        train_data_inputs = train_data_inputs.reshape((N, P, model.input_size))
        #train_data_target = train_data_inputs.reshape((N, P, model.input_size))

    # Train -  Validation split
    tr_inputs, tr_targets, val_inputs, val_targets = train_validation_split(
        train_data_inputs, train_data_targets, tr_split=tr_to_val_split)

    tr_losses, val_losses, model = train_rnn(model=model, nepochs=model.num_epochs, tr_inputs=tr_inputs, tr_targets=tr_targets, 
                                            val_inputs=val_inputs, val_targets=val_targets, tr_verbose=tr_verbose)
    
    if tr_verbose == True:
        plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=True)

    # Trying to visualise training data predictions
    #predictions_rnn_train = predict_rnn(model=model, eval_input=train_data_inputs[0, :, :].reshape((1, P, -1)), n_predict=len(train_data_targets))
    #plot_training_predictions(ytrain=train_data_targets, predictions=predictions_rnn_train, title="Predictions for Training data")

    if len(test_data) > 0:
        predictions_rnn = predict_rnn(model=model, eval_input=train_data_inputs[-1, :, :].reshape((1, P, -1)), n_predict=len(test_data))
        test_error = mean_squared_error(y_true=test_data[:, -1], y_pred=predictions_rnn)
    else:
        #NOTE: Heuristically setting the number of future predictions
        predictions_rnn = predict_rnn(model=model, eval_input=train_data_inputs[-1, :, :].reshape((1, P, -1)), n_predict=132)
        test_error = np.nan # No reference to compare for genearting Test error

    tr_error = tr_losses[-1] # latest training error
    val_error = val_losses[-1] # latest validation error
    #print("**********************************************************************************************************")
    print("{} - {}, {} - {},  {} - {},  {}, - {}".format("Model", model.model_type,
                                                                "Training Error",
                                                                tr_error,
                                                                "Validation Error",
                                                                val_error,
                                                                "Test Error",
                                                                test_error))
    print("***********************************************************************************************************")
    return predictions_rnn, test_error, val_error, tr_error

def train_model_RNN(options, model_type, data, minimum_idx, predict_cycle_num, tau=1, output_file=None, use_grid_search=0, Xmax=None, Xmin=None):
    
    #tau_chosen = 1 #Usual case
    #tau_chosen = options[model_type]["output_size"]
    #print("Tau chosen {}".format(tau_chosen))

    # In case parameter tuning is not carried out
    if use_grid_search == 0:

        # Load the model with the corresponding options
        model = load_model_with_opts(options, model_type)
        
        #NOTE: Obtain the data and targets by heuristically setting p
        num_taps_rnn = 22
        
        X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau = tau, p=num_taps_rnn)

        # Get xtrain, ytrain, ytest
        xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)

        # Pred of q values
        predictions_rnn, test_error, val_error, tr_error = train_and_predict_RNN(model, xtrain, ytrain, ytest, 
                                                                                tr_to_val_split=0.90, tr_verbose=False)
        if len(ytest) > 0:
            
            # Normalized predictions in [0, 1]
            #plot_predictions(predictions=predictions_rnn, ytest=ytest, title="{} model predictions with {} taps for cycle index {}".format(
            #    model_type, num_taps_rnn, predict_cycle_num))
            
            # Unnormalized predictions in original scale
            #ytest_un = np.copy(ytest)
            #ytest_un[:,-1] = unnormalize(ytest[:,-1], Xmax, Xmin)
            #plot_predictions(predictions=unnormalize(predictions_rnn, Xmax, Xmin), ytest=ytest_un, title="{} model predictions (unnormalized) with {} taps for cycle index {}".format(
            #    model_type, num_taps_rnn, predict_cycle_num))
            
            # Save prediction results in a txt file
            save_pred_results(output_file=output_file, predictions=predictions_rnn, te_data_signal=ytest[:,-1])
        else:
            
            #plot_future_predictions(data=data, minimum_idx=minimum_idx, ytrain=ytrain, predictions=predictions_rnn,
            #title="Plot of original timeseries and future predictions for {} for cycle index {}".format(
            #    model_type, predict_cycle_num))
            
            #plot_future_predictions(data=unnormalized_data, minimum_idx=minimum_idx, ytrain=ytrain, predictions=unnormalize(predictions_rnn, Xmax, Xmin),
            #title="Plot of original unnormalized timeseries and future predictions for {} for cycle index {}".format(
            #    model_type, predict_cycle_num))
            
            # Save prediction results in a txt file
            save_pred_results(output_file=output_file, predictions=predictions_rnn, te_data_signal=ytest)

    elif use_grid_search == 1:
        
        orig_stdout = sys.stdout
        f_tmp = open('{}_gs_cycle_{}_logs.txt'.format(model_type, predict_cycle_num), 'a')
        sys.stdout = f_tmp
        gs_params = {"n_hidden":[20, 30, 40, 50, 60]
                    }
        
        gs_list_of_options = create_list_of_dicts(options=options,
                                                model_type=model_type,
                                                param_dict=gs_params)
        
        print("Grid Search to be carried over following {} configs:\n".format(len(gs_list_of_options)))
        val_errors_list = []

        for i, gs_option in enumerate(gs_list_of_options):
            
            print("Config:{} is \n{}".format(i+1, gs_option))
            # Load the model with the corresponding options
            model = RNN_model(
                    input_size=gs_option["input_size"],
                    output_size=gs_option["output_size"],
                    n_hidden=gs_option["n_hidden"],
                    n_layers=gs_option["n_layers"],
                    num_directions=gs_option["num_directions"],
                    model_type=gs_option["model_type"],
                    batch_first=gs_option["batch_first"],
                    lr=gs_option["lr"],
                    device=gs_option["device"],
                    num_epochs=gs_option["num_epochs"],
                )
            
            #NOTE: Obtain the data and targets by heuristically setting p
            num_taps_rnn = 22

            X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau = tau, p=num_taps_rnn)

            # Get xtrain, ytrain, ytest
            xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)

            # Pred of q values
            predictions_rnn, _, val_error, tr_error = train_and_predict_RNN(model, xtrain, ytrain, ytest, 
                                                                                    tr_to_val_split=0.90, 
                                                                                    tr_verbose=True)
            gs_option["Validation_Error"] = val_error
            gs_option["Training_Error"] = tr_error

            val_errors_list.append(gs_option)
            
        with open('gsresults_{}_cycle{}.json'.format(model.model_type, predict_cycle_num), 'w') as f:
            f.write(json.dumps(val_errors_list, indent=2))

        sys.stdout = orig_stdout
        f.close()

    return predictions_rnn
