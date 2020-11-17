import numpy as np
import matplotlib.pyplot as plt

def detect_cycle_minimums(solar_data, window_length, resolution=12, num_precision=3):
    
    solar_data_mean_norm = np.copy(solar_data)
    #solar_data_grad[:, 0] = solar_data[:, 0]
    shifting_factor = np.mean(solar_data[:, 1])
    solar_data_mean_norm[:, 1] = solar_data[:, 1] - shifting_factor
    print(len(solar_data))
    solar_data_mean_norm_neg = np.copy(solar_data_mean_norm)
    solar_data_mean_norm_neg[:,1] = np.where(solar_data_mean_norm[:,1] < 0, solar_data_mean_norm[:,1], 0)
    print(len(solar_data_mean_norm_neg))
    #window_length = 11 * 12
    
    # Obtain the minimum values of the time-series
    start = 0
    end = window_length + 1
    num_windows = solar_data_mean_norm_neg.shape[0] // window_length
    # Pad Signal to make sure that all frames have equal number of samples 
    # without truncating any samples from the original signal
    pad_signal_length = num_windows * window_length + window_length
    z = np.zeros((pad_signal_length - solar_data_mean_norm_neg.shape[0]))
    solar_data_mean_norm_neg_pd = np.append(solar_data_mean_norm_neg[:,1], z) 
    
    num_windows = solar_data_mean_norm_neg_pd.shape[0] // window_length
    print("Number of windows: {}".format(num_windows))
    minimum_values = np.zeros((num_windows, 2))
    solar_cycle_ind_array = np.zeros((num_windows, 1))
    #minimum_values[0,0] = solar_data_mean_norm_neg[0,0]
    
    for i in range(num_windows):
        window = solar_data_mean_norm_neg_pd[start:end]
        if np.min(window) < 0:
            minimum_values[i, 0] = (np.argmin(window) / resolution) + (start / resolution) + solar_data_mean_norm[0,0]
            #minimum_values[i, 0] = np.argmin(window)
            minimum_values[i, 1] = np.min(window) + shifting_factor
        start = start + window_length
        end = end + window_length
        solar_cycle_index = np.argwhere((np.around(solar_data[:, 1], num_precision) == np.around(minimum_values[i, 1], 
        decimals=num_precision)))
        solar_cycle_index = solar_cycle_index[np.logical_or((solar_cycle_index<start),(solar_cycle_index<end))]
        print("Start window:{}, End window:{}, Solar data index:{}, Time index:{:.3f}, Min.value:{:.3f} ".format(start,
        end, solar_cycle_index, minimum_values[i, 0], minimum_values[i, 1]))
        solar_cycle_ind_array[i] = solar_cycle_index[-1]
    
    plot_timeseries_with_cycle_minima(solar_data=solar_data, 
                    solar_data_mean_norm=solar_data_mean_norm,
                    solar_data_mean_norm_neg=solar_data_mean_norm_neg,
                    minimum_values=minimum_values)
    
    return minimum_values, solar_cycle_ind_array

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

def split_tr_test_data(time_indices, solar_data, cycle_indices, 
                       specific_cycle_number=None):

    index_cycle_start = cycle_indices[specific_cycle_number-1, 1]
    index_cycle_end = cycle_indices[specific_cycle_number-1, 2]
    traindata_time = time_indices[0:index_cycle_start]
    traindata_solar = solar_data[0:index_cycle_start]
    testdata_time = time_indices[index_cycle_start:index_cycle_end]
    testdata_solar = solar_data[index_cycle_start:index_cycle_end]

    return traindata_time, traindata_solar, testdata_time, testdata_solar

def plot_train_test_data(trdata_time, trdata_signal, tedata_time, tedata_signal):
    
    # the function returns you the training and test signal as numpy arrays
    plt.figure()
    plt.plot(trdata_time, trdata_signal, 'r+-', linewidth=3)
    plt.plot(tedata_time, tedata_signal, 'b+-', linewidth=3)
    plt.title("Plot of training and testing data", fontsize=20)
    plt.xlabel('Time (in months)', fontsize=16)
    plt.ylabel('Signal Amplitude', fontsize=16)
    plt.legend(["train_signal","test_signal"])
    plt.show()
    return None

def get_train_test_realsolar(datafile):

    real_solar_data = np.loadtxt(datafile)
    print("The shape of the solar data is:{}".format(real_solar_data.shape))

    # Obtaining and plotting data for "Solar data"
    
    print("************* Solar data *************")
    window_length = 11 * 12
    minimum_values_real, solar_cycle_ind_array_real = detect_cycle_minimums(solar_data=real_solar_data, 
    window_length=window_length, resolution=12, num_precision=3)
    cycle_indices_real = np.array([[i+1, int(solar_cycle_ind_array_real[i]), int(solar_cycle_ind_array_real[i+1])] 
        for i in range(0, len(solar_cycle_ind_array_real) - 1)])
    print("Cycle indices of shape {} (<cycle no., start, end>) are:\n{}".format(cycle_indices_real.shape,cycle_indices_real))
    
    trdata_time_real, trdata_signal_real, tedata_time_real, tedata_signal_real = split_tr_test_data(time_indices=real_solar_data[:, 0],
        solar_data=real_solar_data[:,1], cycle_indices=cycle_indices_real, specific_cycle_number=24)

    #plot_train_test_data(trdata_time_real, trdata_signal_real, tedata_time_real, tedata_signal_real)
    return trdata_time_real, trdata_signal_real, tedata_time_real, tedata_signal_real

def main():

    # Importing and loading the data
    dyn_data = np.loadtxt("./data/dynamo_esn.txt")
    real_solar_data = np.loadtxt("./data/solar_data.txt")
    #print("First few points of Dynamo data set:{}".format(dyn_data[:2650,:]))
    #print("First few points of Solar data set:{}".format(real_solar_data[:500,:]))
    print("The shape of the Dynamo data is:{}".format(dyn_data.shape))
    print("The shape of the solar data is:{}".format(real_solar_data.shape))

    # Obtaining and plotting data for "Real Solar data"
    
    print("************* Solar data *************")
    window_length = 11 * 12
    minimum_values_real, solar_cycle_ind_array_real = detect_cycle_minimums(solar_data=real_solar_data, 
    window_length=window_length, resolution=12, num_precision=3)
    cycle_indices_real = np.array([[i+1, int(solar_cycle_ind_array_real[i]), int(solar_cycle_ind_array_real[i+1])] 
        for i in range(0, len(solar_cycle_ind_array_real) - 1)])
    print("Cycle indices of shape {} (<cycle no., start, end>) are:\n{}".format(cycle_indices_real.shape,cycle_indices_real))
    
    trdata_time_real, trdata_signal_real, tedata_time_real, tedata_signal_real = split_tr_test_data(time_indices=real_solar_data[:, 0],
        solar_data=real_solar_data[:,1], cycle_indices=cycle_indices_real, specific_cycle_number=24)

    plot_train_test_data(trdata_time_real, trdata_signal_real, tedata_time_real, tedata_signal_real)
    '''
    # Obtaining and plotting data for "Synthetic Solar data"
    print("************* Dynamo data *************")
    window_length = 11 * 13
    minimum_values_dyn, solar_cycle_ind_array_dyn = detect_cycle_minimums(solar_data=dyn_data, 
    window_length=window_length, resolution=10, num_precision=7)
    cycle_indices_dyn = np.array([[i+1, int(solar_cycle_ind_array_dyn[i]), int(solar_cycle_ind_array_dyn[i+1])] 
        for i in range(0, len(solar_cycle_ind_array_dyn) - 1)])
    print("Cycle indices of shape {} (<cycle no., start, end>) are:\n{}".format(cycle_indices_dyn.shape,cycle_indices_dyn))
    #trdata_time_dyn, trdata_signal_dyn, tedata_time_dyn, tedata_signal_dyn = split_tr_test_data(time_indices=dyn_data[:, 0],
    #    solar_data=dyn_data[:,1], cycle_indices=cycle_indices_dyn, specific_cycle_number=53)

    #plot_train_test_data(trdata_time_dyn, trdata_signal_dyn, tedata_time_dyn, tedata_signal_dyn)
    '''
    return None

if __name__ == "__main__":
    
    main()
    

    