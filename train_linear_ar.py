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
from armodel import create_dataset, train_armodel, predict_armodel
from load_model import load_model_with_opts

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
        xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)
        options[model_type]["num_taps"] = p
        model = load_model_with_opts(options, model_type)
        predictions, test_error, val_error, tr_error = train_and_predict_AR(
            model, xtrain, ytrain, ytest)

        val_errors[:, i] = val_error
        training_errors[:, i] = tr_error
        test_errors[:, i] = test_error

    return training_errors, val_errors, test_errors

def train_and_predict_AR(model, train_data_inputs, train_data_targets, test_data, tr_to_val_split=0.9, tr_verbose=False):
    
    # Count number of model parameters
    total_num_params, total_num_trainable_params = count_params(model=model)
    print("The total number of params: {} and the number of trainable params:{}".format(total_num_params, total_num_trainable_params))
    
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
    with open("results_{}.txt".format(model_type), "a") as fp:
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

def train_model_AR(options, model_type, data, minimum_idx, predict_cycle_num, tau=1, output_file=None, use_grid_search=0):

    # Load the model with corresponding options
    if use_grid_search == 0:
        
        model = load_model_with_opts(options, model_type)
        X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau=1, p=options[model_type]["num_taps"])
        # predict cycle index = entered predict cycle num - 1
        xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)
        # pred of q values
        predictions_ar, test_error, val_error, tr_error = train_and_predict_AR(model, xtrain, ytrain, ytest, tr_to_val_split=0.9, tr_verbose=True)
        
        # Save prediction results in a txt file
        if len(ytest) > 0:
            plot_predictions(predictions=predictions_ar, ytest=ytest, title="AR model predictions with {} taps for cycle index {}".format(
            options[model_type]["num_taps"], predict_cycle_num))
            save_pred_results(output_file=output_file, predictions=predictions_ar, te_data_signal=ytest[:,-1])

    elif use_grid_search == 1:
        
        logfile = './param_selection/{}_gs_cycle_{}_logs.txt'.format(model_type, predict_cycle_num)
        
        orig_stdout = sys.stdout
        f_tmp = open(logfile, 'w')
        sys.stdout = f_tmp

        Error_dict = {}
        test_predictions = []
        test_error_optimal = []
        #nval = 1
        num_total_cycles = len(np.diff(minimum_idx))
        #predict_cycle_num_array = list(np.arange(num_total_cycles-nval, num_total_cycles))
        predict_cycle_num_array = [predict_cycle_num]
        params = {"num_taps":list(np.arange(10, 40, 1))} # For Dynamo
        #params = {"num_taps":list(np.arange(5, 50, 2))} # For Solar
        #TODO: Fix array nature of optimal_num_taps_all
        optimal_num_taps_all, training_errors_all, val_errors_all, test_errors_all = grid_search_AR_all_cycles(data=data,
            solar_indices=minimum_idx, model_type=model_type, options=options, params=params, predict_cycle_num_array=predict_cycle_num_array)
        
        Error_dict["validation_errors_with_taps"] = [(float(params["num_taps"][i]), *val_errors_all[:,i]) 
            for i in range(val_errors_all.shape[1])]

        plt.figure()
        plt.plot(params["num_taps"], val_errors_all[0], label="Validation MSE")
        plt.ylabel("MSE")
        plt.xlabel("Number of taps")
        plt.legend()
        plt.title("Error (MSE) vs number of taps")
        plt.show()

        if type(optimal_num_taps_all) != list:
            optimal_num_taps_all = [optimal_num_taps_all]

        Error_dict["optimal_num_taps"] = [float(*optimal_num_taps_all)] #NOTE: Object of int64 is not json serializable

        # Retrain the model again with the optimal value
        for i, optimal_num_taps in enumerate(optimal_num_taps_all):
            
            options[model_type]["num_taps"] = optimal_num_taps
            model = load_model_with_opts(options, model_type)
            X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx,tau=1, p=optimal_num_taps)
            xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num_array[i])
            # pred of q values
            predictions_ar, test_error, val_error, tr_error = train_and_predict_AR(model, xtrain, ytrain, ytest, 
                tr_to_val_split=0.90, tr_verbose=True)
            test_predictions.append(predictions_ar.tolist())
            if len(ytest) > 0:
                
                plot_predictions(predictions=predictions_ar, ytest=ytest, title="AR model predictions with {} taps for cycle index {}".format(
                    optimal_num_taps, predict_cycle_num_array[i]))
                test_error_optimal.append(test_error)
            
            else:
                
                plot_future_predictions(data=data, minimum_idx=minimum_idx, ytrain=ytrain,
                                        predictions=predictions_ar, title='Plot of original timeseries and future predictions for AR model')

        Error_dict["Test_predictions"] = test_predictions
        if len(test_error_optimal) > 0:
            Error_dict["Test_error"] = [test_error_optimal]
        else:
            Error_dict["Test_error"] = []

        with open('./param_selection/gsresults_{}_cycle{}.json'.format(model_type, predict_cycle_num_array[i]), 'w+') as fp:
            json.dump(Error_dict, fp, indent=2)
        
        # Saving result files properly
        if len(ytest) > 0:
            save_pred_results(output_file=output_file, predictions=predictions_ar, te_data_signal=ytest[:,-1])

    return predictions_ar
