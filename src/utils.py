import numpy as np
import sys
import itertools

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
    return X_norm

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
        _, minimum_idx = detect_cycle_minimums_solar(solar_data=X, window_length=11 * 12, resolution=12,
                                                     num_precision=3)
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

    # plot_timeseries_with_cycle_minima(solar_data=solar_data,
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
    return np.concatenate([xx[:, col].reshape(1, -1) for xx in x], axis=0)


def get_cycle(X, Y, icycle):
    """ Retrives the training data, training targets and test targets for 
    predicting the i-th cycle
    Args:
        X ([list of list of tuples]): training data + targets (training_data_i, training_target_i) 
            for all solar cycles (including future cycle, for which test data is not available yet)
        Y ([list]): Data for test cycles 
        icycle ([int]): cycle index

    Returns:
        xtrain - For predicting the i-th cycle, the number of 'p'-shifted training data 
        points (contains data upto i-th cycle, but not including it)
        ytrain - For predicting the i-th cycle, the number of 'p'-shifted training data targets
        (contains data upto i-th cycle, but not including it)
        ytest - The data points corresponding to the i-th cycle 
    """
    if icycle == len(X):
        ytest=np.array([])
        tmp = sum(X[:icycle+1], [])
        xtrain = [t[0] for t in tmp]
        ytrain = [t[1] for t in tmp]
    else:
        ytest = Y[icycle]
        tmp = sum(X[:icycle+1], []) # Aggregates all the training data upto the i-th cycle
        xtrain = [t[0] for t in tmp] 
        ytrain = [t[1] for t in tmp]
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

    print("Grid-search will be computed for the following set of parameter lists:\n{}".format(len(params_dict_list_all)))
    return params_dict_list_all