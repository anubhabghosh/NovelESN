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
    
    elif model_type == "linear_ar":

        model = Linear_AR(
                        num_taps=options[model_type]["num_taps"],
                        lossfn_type=options[model_type]["lossfn_type"],
                        lr=options[model_type]["lr"],
                        num_epochs=options[model_type]["num_epochs"],
                        init_net=options[model_type]["init_net"],
                        device=options[model_type]["device"] 
        )
    elif model_type in ["rnn", "lstm", "gru"]:

        model = RNN_model(
                        input_size=options[model_type]["input_size"],
                        output_size=options[model_type]["output_size"],
                        n_hidden=options[model_type]["n_hidden"],
                        n_layers=options[model_type]["n_layers"],
                        num_directions=options[model_type]["num_directions"],
                        model_type=options[model_type]["model_type"],
                        batch_first=options[model_type]["batch_first"],
                        lr=options[model_type]["lr"],
                        device=options[model_type]["device"],
                        num_epochs=options[model_type]["num_epochs"],
        )
    
    return model

def train_and_predict_ESN(model, train_data, test_data=None):
    
    tr_data_signal = train_data[:, -1].reshape((-1, 1))
    te_data_signal = test_data[:, -1].reshape((-1, 1))
    
    model.teacher_forcing(tr_data_signal)
    train_mse = model.train(train_signal=tr_data_signal)
    print("training mse over q =", train_mse / model.yntau_size_q)
    predictions, pred_indexes = model.predict()
    test_mse = mean_squared_error(te_data_signal, predictions)
    print("test mse=", test_mse)
    
    plot_predictions(ytest=test_data,
                    predictions=predictions,
                    title="Predictions using NovelESN model")

    return predictions, te_data_signal, pred_indexes


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

#TODO: Implement the version here
def train_and_predict_RNN(model, train_data_inputs, train_data_targets, test_data, tr_to_val_split=0.9, tr_verbose=False):

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

def plot_training_predictions(ytrain, predictions, title):

    #Prediction plot
    plt.figure()
    #plt.title("Prediction value of number of sunspots vs time index", fontsize=20)
    plt.title(title, fontsize=10)
    plt.plot(ytrain, label="actual training signal", color="orange")
    plt.plot(predictions, label="prediction", color="green")
    plt.legend()
    plt.show()

def plot_predictions(ytest, predictions, title):

    #Prediction plot
    plt.figure()
    #plt.title("Prediction value of number of sunspots vs time index", fontsize=20)
    plt.title(title, fontsize=10)
    plt.plot(ytest[:,0], ytest[:,1], '+-', label="actual test signal", color="orange")
    plt.plot(ytest[:,0], predictions, '*-', label="prediction", color="green")
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

def plot_future_predictions(data, minimum_idx, ytrain, predictions, title=None):
    
    resolution = np.around(np.diff(data[:,0]).mean(),1)
    plt.figure()
    plt.plot(data[:minimum_idx[-1],0], data[:minimum_idx[-1],1], 'r+-')
    plt.plot(np.arange(ytrain[-1][-1][0] + resolution, ((len(predictions)) * resolution) + 
        ytrain[-1][-1][0], resolution), predictions, 'b*-')
    plt.legend(['Original timeseries', 'Future prediction'])
    if title is None:
        plt.title('Plot of original timeseries and future predictions')
    else:
        plt.title(title)
    plt.show()

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

def main():
    
    parser = argparse.ArgumentParser(description=
    "Use a variety of recurrent architectures for predicting solar sunpots as a time series\n"\
    "Example: python main_gs.py --model_type [esn/linear_ar/rnn/lstm/gru] --dataset dynamo --train_file [full path to training data file] \
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
    parser.add_argument("--predict_cycle_num", help="Cycle index to be predicted", default=None, type=int)
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
    #data[:, 1] = np.diff(data[:,1], prepend=data[0, 1])

    # Get multiple step ahead prediction datasets : #NOTE: Only for Linear_AR so far
    if model_type == "esn":
        
        X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau=1, 
            p=1)
        
        # predict cycle index = entered predict cycle num - 1
        xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)

        options["esn"]["tau"] = len(ytest) - 1
        options["esn"]["history_q"] = options["esn"]["tau"] + 1
        model = load_model_with_opts(options, model_type)

        # Concat data 
        xtrain_ct = concat_data(xtrain, col=-1)
        ytrain_ct = concat_data(ytrain, col=-1)

        #tr_data_signal = xtrain_ct[:, -1].reshape((-1, 1))
        #te_data_signal = ytest[:, -1].reshape((-1, 1))

        # pred of q values
        predictions, te_data_signal, pred_indexes = train_and_predict_ESN(model, train_data=xtrain_ct, test_data=ytest)
        
        # Saving prediction results
        save_pred_results(output_file=output_file, predictions=predictions, te_data_signal=te_data_signal)

    elif model_type == "linear_ar":

        # Load the model with corresponding options
        if use_grid_search == 0:
            
            model = load_model_with_opts(options, model_type)
            X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau=1, p=options[model_type]["num_taps"])
            # predict cycle index = entered predict cycle num - 1
            xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)
            # pred of q values
            predictions_ar, test_error, val_error, tr_error = train_and_predict_AR(model, xtrain, ytrain, ytest, tr_to_val_split=0.9, tr_verbose=True)
            plot_predictions(predictions=predictions_ar, ytest=ytest, title="AR model predictions with {} taps for cycle index {}".format(
                options[model_type]["num_taps"], predict_cycle_num))
            
            # Save prediction results in a txt file
            save_pred_results(output_file=output_file, predictions=predictions_ar, te_data_signal=ytest[:,-1])

        elif use_grid_search == 1:
            
            Error_dict = {}
            test_predictions = []
            test_error_optimal = []
            nval = 1
            num_total_cycles = len(np.diff(minimum_idx))
            #predict_cycle_num_array = list(np.arange(num_total_cycles-nval, num_total_cycles))
            predict_cycle_num_array = [predict_cycle_num]
            params = {"num_taps":list(np.arange(10, 50, 2))} # For Dynamo
            #params = {"num_taps":list(np.arange(5, 50, 2))} # For Solar
            #TODO: Fix array nature of optimal_num_taps_all
            optimal_num_taps_all, training_errors_all, val_errors_all, test_errors_all = grid_search_AR_all_cycles(data=data,
                solar_indices=minimum_idx, model_type=model_type, options=options, params=params, predict_cycle_num_array=predict_cycle_num_array)
            
            Error_dict["validation_errors_with_taps"] = [(float(params["num_taps"][i]), *val_errors_all[:,i]) 
                for i in range(val_errors_all.shape[1])]


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

            Error_dict["optimal_num_taps"] = [float(*optimal_num_taps_all)] #NOTE: Object of int64 is not json serializable

            # Retrain the model again with the optimal value
            for i, optimal_num_taps in enumerate(optimal_num_taps_all):
                
                options[model_type]["num_taps"] = optimal_num_taps
                model = load_model_with_opts(options, model_type)
                X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx,tau=1, p=optimal_num_taps)
                xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num_array[i])
                # pred of q values
                predictions_ar, test_error, val_error, tr_error = train_and_predict_AR(model, xtrain, ytrain, ytest, tr_to_val_split=0.75, tr_verbose=True)
                test_predictions.append(predictions_ar.tolist())
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
            if len(test_error_optimal) > 0:
                Error_dict["Test_error"] = [test_error_optimal]
            else:
                Error_dict["Test_error"] = []

            with open('./log/grid_search_results_{}_cycle{}.json'.format(dataset, predict_cycle_num_array[i]), 'w+') as fp:
                json.dump(Error_dict, fp, indent=2)
            
            #TODO: To fix saving result files properly
            save_pred_results(output_file=output_file, predictions=predictions_ar, te_data_signal=ytest[:,-1])

    elif model_type in ["rnn", "lstm", "gru"]:
       
        # In case parameter tuning is not carried out
        if use_grid_search == 0:

            # Load the model with the corresponding options
            model = load_model_with_opts(options, model_type)
            
            #NOTE: Obtain the data and targets by heuristically setting p
            num_taps_rnn = 22
            X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau = 1, p=num_taps_rnn)

            # Get xtrain, ytrain, ytest
            xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)

            # Pred of q values
            predictions_rnn, test_error, val_error, tr_error = train_and_predict_RNN(model, xtrain, ytrain, ytest, 
                                                                                    tr_to_val_split=0.90, tr_verbose=True)
            if len(ytest) > 0:
                
                # Normalized predictions in [0, 1]
                plot_predictions(predictions=predictions_rnn, ytest=ytest, title="{} model predictions with {} taps for cycle index {}".format(
                    model_type, num_taps_rnn, predict_cycle_num))
                
                # Unnormalized predictions in original scale
                ytest_un = np.copy(ytest)
                ytest_un[:,-1] = unnormalize(ytest[:,-1], Xmax, Xmin)
                plot_predictions(predictions=unnormalize(predictions_rnn, Xmax, Xmin), ytest=ytest_un, title="{} model predictions (unnormalized) with {} taps for cycle index {}".format(
                    model_type, num_taps_rnn, predict_cycle_num))

                # Save prediction results in a txt file
                save_pred_results(output_file=output_file, predictions=predictions_rnn, te_data_signal=ytest[:,-1])
            else:

                plot_future_predictions(data=data, minimum_idx=minimum_idx, ytrain=ytrain, predictions=predictions_rnn,
                title="Plot of original timeseries and future predictions for {} for cycle index {}".format(
                    model_type, predict_cycle_num))
                
                plot_future_predictions(data=unnormalized_data, minimum_idx=minimum_idx, ytrain=ytrain, predictions=unnormalize(predictions_rnn, Xmax, Xmin),
                title="Plot of original unnormalized timeseries and future predictions for {} for cycle index {}".format(
                    model_type, predict_cycle_num))
                
                # Save prediction results in a txt file
                save_pred_results(output_file=output_file, predictions=predictions_rnn, te_data_signal=ytest)

        elif use_grid_search == 1:
            
            gs_params = {"n_hidden":[30, 40, 50]
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
                X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau = 1, p=num_taps_rnn)

                # Get xtrain, ytrain, ytest
                xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)

                # Pred of q values
                predictions_rnn, _, val_error, tr_error = train_and_predict_RNN(model, xtrain, ytrain, ytest, 
                                                                                        tr_to_val_split=0.90, 
                                                                                        tr_verbose=True)
                gs_option["Validation_Error"] = val_error
                gs_option["Training_Error"] = tr_error

                val_errors_list.append(gs_option)
                
            with open('gsresults_cycle{}.json'.format(predict_cycle_num), 'w') as f:
                f.write(json.dumps(val_errors_list, indent=2))



if __name__ == "__main__":
    main()

