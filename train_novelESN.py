# Import the necessary libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_msah_training_dataset, get_minimum, concat_data, get_cycle, plot_predictions, plot_future_predictions
from src.utils import normalize, create_list_of_dicts, save_pred_results, unnormalize, plot_losses
import torch
import copy
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import json
import itertools
import argparse
from load_model import load_model_with_opts

def train_and_predict_ESN(model, train_data, test_data=None):
    
    if len(test_data) > 0:
        tr_data_signal = train_data[:, -1].reshape((-1, 1))
        te_data_signal = test_data[:, -1].reshape((-1, 1))
        print(tr_data_signal.shape)
        model.teacher_forcing(tr_data_signal)
        train_mse = model.train(train_signal=tr_data_signal)
        #print("training mse over q =", train_mse / model.yntau_size_q)
        predictions, pred_indexes = model.predict()
        test_mse = mean_squared_error(te_data_signal, predictions)
        #print("test mse=", test_mse)
        print("{} - {},  {} - {:.8f},  {}, - {:.8f}".format("Model", "ESN",
                                                            "Training Error",
                                                            train_mse,
                                                            "Test Error",
                                                            test_mse))
        print("***********************************************************************************************************")
    
    else:

        tr_data_signal = train_data[:, -1].reshape((-1, 1))
        print(tr_data_signal.shape)
        te_data_signal = None
        #te_data_signal = test_data[:, -1].reshape((-1, 1))
        model.teacher_forcing(tr_data_signal)
        train_mse = model.train(train_signal=tr_data_signal)
        #print("training mse over q =", train_mse / model.yntau_size_q)
        predictions, pred_indexes = model.predict()
        test_mse = np.nan
        #print("test mse=", test_mse)
        print("{} - {},  {} - {:.8f},  {}, - {:.8f}".format("Model", "ESN",
                                                            "Training Error",
                                                            train_mse,
                                                            "Test Error",
                                                            test_mse))
        print("***********************************************************************************************************")
    
    #plot_predictions(ytest=test_data,
    #                predictions=predictions,
    #                title="Predictions using NovelESN model")

    return predictions, te_data_signal, pred_indexes

def train_model_ESN(options, model_type, data, minimum_idx, predict_cycle_num, tau=1, output_file=None):

    # Get the dataset of inputs and targets based on num_taps
    if predict_cycle_num == 23 or predict_cycle_num == 76:
        X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau=1, p=1)
        # predict cycle index = entered predict cycle num - 1
        xtrain, ytrain, ytest = get_cycle(X, Y, icycle=predict_cycle_num)
        #print(ytest)
        options["esn"]["tau"] = 132
        options["esn"]["history_q"] = options["esn"]["tau"] + 1
        model = load_model_with_opts(options, model_type)

        # Concat data 
        print(xtrain[1].shape)
        xtrain_ct = concat_data(xtrain, col=-1)
        #ytrain_ct = concat_data(ytrain, col=None)
        print(xtrain_ct.shape)

    else:
        X, Y = get_msah_training_dataset(data, minimum_idx=minimum_idx, tau=1, p=1)
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
    if len(ytest) > 0:
        save_pred_results(output_file=output_file, predictions=predictions, te_data_signal=te_data_signal)

    return predictions, ytest