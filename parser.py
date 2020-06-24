#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:42:10 2020

@author: Aleix Espuña Fontcuberta
"""
import argparse

def define_parser():

    descrip = "program that predicts a time series of scalars at certain"\
    " instants of time specified by the user."

    parser = argparse.ArgumentParser(description=descrip)
    parser.add_argument("training_file", type=str, help="path to a txt file" 
    " with a single column representing the training time series "
    "y_0, y_1, ...y_M-1.") 
    
    parser.add_argument("output_file", type=str, help="path to the output file"
                                      " where the prediction will be written.")
    
    parser.add_argument("tau", type=int, help="Size of the jump in the future."
    " Given a training data of M scalars y_0, y_1, ...y_M-1, tau indicates"
    " that the last predicted value will be y_M+tau.")
    
    parser.add_argument("history_q", type=int, help="Determines the amount of "
    "predicted values and the beginning of the prediction. The total number of"  
    " predicted values will be q and they will go from instant "
    "M+tau-(q-1) to M+tau), generating the predictions "
    "y_M+tau-(q-1), y_M+tau-(q-2) ... y_M+tau.")
    
    parser.add_argument("-beta", type=float, default=1e-10, nargs="?", 
    help="Regularization parameter for a ridge regression equation."
    "See the file equations_doc.pdf for more info")
    
    parser.add_argument("-test_file", type=str, default=None, help="Path to a "
    "txt file with test data values. The test data is used to know if the "
    "prediction is good or not. A perfect prediction should match the test "
    "data. If provided, the program will show a plot comparing the prediction"
    "and the test data.", nargs="?")
    
    parser.add_argument("-history_p", type=int, default=32, help="Number of "
    "feedback values used to update the reservoir of neurons during training."
    " See equations_doc.pdf for more info", nargs="?")
    
    return parser


