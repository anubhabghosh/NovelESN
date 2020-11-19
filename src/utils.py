import numpy as np
import sys


def get_minimum(X, dataset):
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
    return index_of_minimums.astype(int).reshape(-1)


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
        tau:         Number of the steps ahead
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
            xtrain.append(X[:minimum_idx[icycle]])

        if icycle + 1 < minimum_idx.shape[0]:
            Y.append(X[minimum_idx[icycle]:minimum_idx[icycle + 1], :])


    return xtrain, Y


def concat_data(x, col=1):
    """Concatenate all the `col` column of the element"""
    return np.concatenate([xx[:, col].reshape(1, -1) for xx in x], axis=0)


def get_cycle(X,Y,icycle):
    ytest = Y[icycle]
    tmp = sum(X[:icycle + 1], [])
    xtrain = [t[0] for t in tmp]
    ytrain = [t[1] for t in tmp]
    return xtrain,ytrain,ytest
