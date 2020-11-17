#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:59:06 2020

@author: aleix
"""

import numpy as np
import matplotlib.pyplot as plt

# THIS SET OF FUNCTIONS ONLY WORK FOR THE DYNAMO SIGNAL, NOT FOR THE SOLAR DATA!

def get_train_test_dynamo(time, dynamo, cycle_num):
    index_minimums = find_index_of_minimums(dynamo) 
    index_cycle_start = index_minimums[cycle_num-1]
    index_cycle_end = index_minimums[cycle_num]
    
    time_train = time[0: index_cycle_start]
    train_signal = dynamo[0: index_cycle_start]
    time_test = time[index_cycle_start: index_cycle_end]
    test_signal = dynamo[index_cycle_start: index_cycle_end]
    return time_train, train_signal, time_test, test_signal


def find_index_of_minimums(dynamo_signal):
    index_of_minimums = []
    for i in range(1, dynamo_signal.size): # point 0 has not a preceeding point
        is_minimum = check_if_is_minimum(dynamo_signal, i)
        if is_minimum:
            index_of_minimums.append(i)
    return index_of_minimums


def check_if_is_minimum(signal, index):
    if signal[index-1] > signal[index] and signal[index+1] > signal[index]:
        is_minium = True
    else:
        is_minium = False
    return is_minium

# additional function that I used to test the function find_index_minimums
def plot_minimums_found(dynamo_signal, index_of_minimums):
    plt.figure()
    plt.plot(dynamo_signal)
    for index in index_of_minimums:
        plt.plot(index, dynamo_signal[index], "go", markersize=3)
    return

def main():

    # Example of selection and plotting of cycle 45
    cycle = 45
    path_to_dynamo_file = "./data/dynamo_esn.txt" # change this according to your path

    dynamo = np.loadtxt(path_to_dynamo_file, usecols=1)
    time = np.loadtxt(path_to_dynamo_file, usecols=0)
    time_train, train, time_test, test = get_train_test_dynamo(time, dynamo, cycle)
    # the function returns you the training and test signal as numpy arrays

    plt.figure()
    plt.plot(time_train, train, label="train_signal")
    plt.plot(time_test, test, label="test_signal")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()