import numpy as np
import sys
from parse import parse
import os
import itertools
import matplotlib.pyplot as plt

def plot_timeseries_with_cycle_minima(solar_data, solar_data_mean_norm, solar_data_mean_norm_neg, minimum_values):
    
    #print("Minimum values with indices are:\n{}".format(minimum_values))
    
    plt.rcParams["figure.figsize"] = (20,10)
    plt.figure()
    plt.plot(solar_data[:,0], solar_data[:,1], 'r+-', linewidth=3)
    plt.plot(solar_data_mean_norm[:,0], solar_data_mean_norm[:, 1], 'b--', linewidth=2)
    plt.plot(solar_data_mean_norm_neg[:,0], solar_data_mean_norm_neg[:, 1], 'k+-', linewidth=2)
    plt.plot(minimum_values[:,0], minimum_values[:,1], 'g*', linewidth=4, markersize=15)
    plt.xlabel('Time (in months)', fontsize=16)
    plt.ylabel('Signal Amplitude', fontsize=16)
    plt.title("Plot of the Solar dataset with Cycle minima", fontsize=20)
    plt.legend(['Original data', 'Mean normalized data', 'Mean normalized data - positive clipped', 'Cycle-wise Minimum values'])
    plt.show()
    return None

def normalize(X, feature_space=(0, 1)):
    """ Normalizing the features in the feature_space (lower_lim, upper_lim)

    Args:
        X ([numpy.ndarray]): Unnormalized data consisting of signal points
        feature_space (tuple, optional): [lower and upper limits]. Defaults to (0, 1).

    Returns:
        X_norm [numpy.ndarray]: Normalized feature values
    """
    X_norm = (X - X.min())/(X.max() - X.min()) * (feature_space[1] - feature_space[0]) + \
        feature_space[0]
    return X_norm, X.max(), X.min()

def unnormalize(X_norm, X_max, X_min, feature_space=(0, 1)):
    X_unnorm = ((X_norm - feature_space[0]) / (feature_space[1] - feature_space[0])) * (X_max - X_min) + \
        X_min
    return X_unnorm

def get_minimum(X, dataset):
    """ This function returns the 'minimum' indices or the indices for
    the solar cycles for the particular dataset type.

    Args:
        X ([numpy.ndarray]): The complete time-series data present as (N_samples x 2),
        with each row being of the form (time-stamp x signal value)
        dataset ([str]): String to indicate the type of the dataset - solar / dynamo

    Returns:
        minimum_idx : An array containing the list of indices for the minimum 
        points of the data
    """
    if dataset == "solar":
        #NOTE: Setting a num_precision of 5 is becoming necessary after normalization of data in [0,1],
        # for unnormalized data, num_precision can also work
        _, minimum_idx = detect_cycle_minimums_solar(solar_data=X, window_length=11 * 12, resolution=12,
                                                     num_precision=5, verbose=False)
    elif dataset == "dynamo":
        minimum_idx = find_index_of_minimums_dyn(X[:, 1])
    else:
        print("Dataset {} unknown".format(dataset))
        sys.exit(1)
    return minimum_idx

def find_index_of_minimums_dyn(dynamo_signal):
    index_of_minimums = []
    for i in range(1, dynamo_signal.size): # point 0 has not a preceeding point
        is_minimum = check_if_is_minimum(dynamo_signal, i)
        if is_minimum:
            index_of_minimums.append(i)
    return np.array(index_of_minimums).astype(int).reshape(-1)


def check_if_is_minimum(signal, index):
    if signal[index-1] > signal[index] and signal[index+1] > signal[index]:
        is_minium = True
    else:
        is_minium = False
    return is_minium


def detect_cycle_minimums_solar(solar_data, window_length, resolution=12, num_precision=3, verbose=False):
    solar_data_mean_norm = np.copy(solar_data)
    # solar_data_grad[:, 0] = solar_data[:, 0]
    shifting_factor = np.mean(solar_data[:, 1])
    solar_data_mean_norm[:, 1] = solar_data[:, 1] - shifting_factor
    if verbose:
        print(len(solar_data))
    solar_data_mean_norm_neg = np.copy(solar_data_mean_norm)
    solar_data_mean_norm_neg[:, 1] = np.where(solar_data_mean_norm[:, 1] < 0, solar_data_mean_norm[:, 1], 0)
    if verbose:
        print(len(solar_data_mean_norm_neg))
    # window_length = 11 * 12

    # Obtain the minimum values of the time-series
    start = 0
    end = window_length + 1
    num_windows = solar_data_mean_norm_neg.shape[0] // window_length
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal_length = num_windows * window_length + window_length
    z = np.zeros((pad_signal_length - solar_data_mean_norm_neg.shape[0]))
    solar_data_mean_norm_neg_pd = np.append(solar_data_mean_norm_neg[:, 1], z)

    num_windows = solar_data_mean_norm_neg_pd.shape[0] // window_length
    if verbose:
        print("Number of windows: {}".format(num_windows))
    minimum_values = np.zeros((num_windows, 2))
    solar_cycle_ind_array = np.zeros((num_windows, 1))
    # minimum_values[0,0] = solar_data_mean_norm_neg[0,0]

    for i in range(num_windows):
        window = solar_data_mean_norm_neg_pd[start:end]
        if np.min(window) < 0:
            minimum_values[i, 0] = (np.argmin(window) / resolution) + (start / resolution) + solar_data_mean_norm[0, 0]
            # minimum_values[i, 0] = np.argmin(window)
            minimum_values[i, 1] = np.min(window) + shifting_factor
        start = start + window_length
        end = end + window_length
        solar_cycle_index = np.argwhere((np.around(solar_data[:, 1], num_precision) == np.around(minimum_values[i, 1],
                                                                                                 decimals=num_precision)))
        solar_cycle_index = solar_cycle_index[np.logical_or((solar_cycle_index < start), (solar_cycle_index < end))]
        if verbose:
            print("Start window:{}, End window:{}, Solar data index:{}, Time index:{:.3f}, Min.value:{:.3f} ".format(
                start,
                end, solar_cycle_index, minimum_values[i, 0], minimum_values[i, 1]))
        solar_cycle_ind_array[i] = solar_cycle_index[-1]

    #plot_timeseries_with_cycle_minima(solar_data=solar_data,
    #                solar_data_mean_norm=solar_data_mean_norm,
    #                solar_data_mean_norm_neg=solar_data_mean_norm_neg,
    #                minimum_values=minimum_values)

    return minimum_values, solar_cycle_ind_array.astype(int).reshape(-1)


def get_msah_training_dataset(X, minimum_idx, tau=1, p=np.inf):
    """Given
        X:           (n_samples,2), time series with timestamps on the first column and values on the second
        minimum_idx: The index on the minimums 
        tau:         Number of the steps ahead (to predict)
        p:           Order of model (number of steps backwards to use in the training data),
       return
        xtrain: List of lists,
          the ith element of `xtrain` is a list containing the training data relative to the prediction of the samples
          in the ith cycle. if p is np.inf, The training data consists in the data up to the start of the ith cycle
        Y: List of np.ndarray, the ith element of `Y` is the raw data of the ith cycle 

       """
    Y = []
    xtrain = []

    for icycle in range(1, minimum_idx.shape[0]):
        tmp = []
        if not np.isinf(p):
            # i spans the indexes of a cycle
            for i in range(minimum_idx[icycle - 1], minimum_idx[icycle]):
                # tmp receives a tuple (training data, target)
                if i - p >= 0:
                    # Append the p points prior to i
                    tmp.append((X[i - p:i, :], X[i:i + tau]))
                else:
                    # Do not append the data segment if it is shorter than p
                    pass

            #tmp.append((X[0:i, :], X[i:i + tau]))
            xtrain.append(tmp)
        else:
            # If p is given as np.inf, in that case the entire signal is used for
            # prediction relative to the target
            xtrain.append(X[:minimum_idx[icycle]])

        if icycle + 1 < minimum_idx.shape[0]:
            Y.append(X[minimum_idx[icycle]:minimum_idx[icycle + 1], :])

    return xtrain, Y

def concat_data(x, col=1):
    """Concatenate all the `col` column of the element"""
    if col == 1:
        return np.concatenate([xx[:, col].reshape(1, -1) for xx in x], axis=0)
    elif col == -1:
        return np.concatenate([xx[:, :].reshape(1, -1) for xx in x], axis=0)

def get_cycle(X, Y, icycle):

    if isinstance(X[0], np.ndarray):
        xtrain = X[icycle]
        ytrain = None
    else:
        tmp = sum(X[:icycle + 1], [])
        xtrain = [t[0] for t in tmp]
        ytrain = [t[1] for t in tmp]

    if icycle == len(X) - 1:
        ytest = np.array([])
    else:
        ytest = Y[icycle]
    return xtrain, ytrain, ytest


def create_combined_param_dict(param_dict):

    keys, values = zip(*param_dict.items())
    param_combinations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_combinations_dict

def create_list_of_dicts(options, model_type, param_dict):
    
    params_dict_list_all = []
    param_combinations_dict = create_combined_param_dict(param_dict)
    for p_dict in param_combinations_dict:
        keys = p_dict.keys()
        tmp_dict = options[model_type]
        for key in keys:
            tmp_dict[key] = p_dict[key]
        params_dict_list_all.append(tmp_dict.copy())

    #print("Grid-search will be computed for the following set of parameter lists:\n{}".format(len(params_dict_list_all)))
    return params_dict_list_all

def save_pred_results(output_file, predictions, te_data_signal):
    
    # Save the results in the output file
    if len(te_data_signal) > 0:
        np.savetxt(fname=output_file,
               X=np.concatenate([predictions.reshape(-1, 1), te_data_signal.reshape(-1, 1)], axis=1)
               )
    else:
        # In case of future predictions only need to save the predictions
        # Assuming output folder is of the form "[output_folder]/[filename].[extension]"
        o_folder, o_filename, ext = parse("{}/{}.{}", output_file)
        output_file_modified = os.path.join(o_folder, (o_filename + "_future" + "." + ext))
        np.savetxt(fname=output_file_modified,
               X = np.array(predictions.reshape(-1, 1))
               #X=np.concatenate([predictions.reshape(-1, 1), te_data_signal.reshape(-1, 1)], axis=1)
               )

def plot_losses(tr_losses, val_losses, logscale=False):
    
    plt.figure()
    if logscale == False:

        plt.plot(tr_losses, 'r+-')
        plt.plot(val_losses, 'b*-')
        plt.xlabel("No. of training iterations")
        plt.ylabel("MSE Loss")
        plt.legend(['Training Set', 'Validation Set'])
        plt.title("MSE loss vs. no. of training iterations")

    elif logscale == True:

        plt.plot(np.log10(tr_losses), 'r+-')
        plt.plot(np.log10(val_losses), 'b*-')
        plt.xlabel("No. of training iterations")
        plt.ylabel("Log of MSE Loss")
        plt.legend(['Training Set', 'Validation Set'])
        plt.title("Log of MSE loss vs. no. of training iterations")
        
    #plt.savefig('./models/loss_vs_iterations.pdf')
    plt.show()

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