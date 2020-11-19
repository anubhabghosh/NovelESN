import numpy as np
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

def split_tr_test_data(time_indices, solar_data, cycle_indices, 
                       specific_cycle_number=None):

    index_cycle_start = cycle_indices[specific_cycle_number-1, 1]
    index_cycle_end = cycle_indices[specific_cycle_number-1, 2]
    traindata_time = time_indices[0:index_cycle_start]
    traindata_solar = solar_data[0:index_cycle_start]
    testdata_time = time_indices[index_cycle_start:index_cycle_end]
    testdata_solar = solar_data[index_cycle_start:index_cycle_end]

    return traindata_time, traindata_solar, testdata_time, testdata_solar

def plot_train_test_data(trdata, tedata):
    
    # the function returns you the training and test signal as numpy arrays
    plt.figure()
    plt.plot(trdata[:,0], trdata[:,1], 'r+-', linewidth=3)
    plt.plot(tedata[:,0], tedata[:,1], 'b+-', linewidth=3)
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

    # plot_train_test_data(trdata_time_real, trdata_signal_real, tedata_time_real, tedata_signal_real)
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
    

    