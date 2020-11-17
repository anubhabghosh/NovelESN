#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    NOVEL ESN
    A novel ESN algorithm for predicting signal continuations.
    The novel algorithm solves the numerical instability issues of the 
    standard one.
     

    Copyright (C) 2020  Saikat Chatterjee and Aleix Espuña Fontcuberta

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from collections import deque
import sys
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class NovelEsn(object):
    """ This class defines the model for Echo State Network

    Args:
        - num_neur (int): The number of neurons in the 'Reservoir'
        - Wfb (numpy.ndarray): Feedback matrix
        - Wres (numpy.ndarray): Reservoir matrix (sparse)
        - tau (int): Number of future steps to predict upto
        - history_q (int): Number of prediction steps
        - history_p (int): The dimension of the feedback vector
        - beta (float): The factor used for L2-regularization in Ridge

    Methods:
        - teacher_forcing(train_data): function to enforce teacher forcing 
        and creating reservoir
        - predict(): function to fit using Ridge regression and output predictions
        and mean squared error (fitting error)
        - buildYntauVectors(self, yntau_size_q):
        - update_state(yn):
        - check_tau_q_and_p(self, yntau_size_q):
        
    """
    def __init__(self, num_neur=1000, conn_per_neur=10,
                spectr_rad=0.8, tau=151, history_q=152, history_p=32,
                beta_regularizer=1e-10):
        #self.yn_size_p = yn_size_p
        self.num_neur = num_neur
        self.xstate = np.zeros(num_neur)
        self.Wfb = build_Wfb(num_neur, history_p, spectr_rad)
        self.Wres = build_reservoir(num_neur, conn_per_neur, spectr_rad)
        self.tau = tau
        #self.history_q = history_q
        self.yntau_size_q = history_q # Refers to the parameter 'q'
        self.yn_size_p = history_p # Refers to the parameter 'p'
        self.beta = beta_regularizer
    
    def teacher_forcing(self, train_signal):
        """ This function generates the set of reservoir states
        and corresponding number of reservoir outputs.

        NOTE: This function doesn't solve any optimization problem

        Args:
            train_signal ([numpy.ndarray]): The time-series for the training data

        Returns:
            Xstates ([numpy.ndarray]) : A numpy array of dimension (N-p x num_neurons), 
            with N = length of training signal, p = length of the feedback vector yn,
            and num_neur = number of neurons in the reservoir

        """
        #we will save the training data for the predict function
        self.train_signal = train_signal  
        self.N = len(train_signal)
        
        # min index to build the first y_{n}
        nstart = self.yn_size_p - 1 
        
        # Initially this deque contains values from index p-1 to p
        yn = build_history_deque(train_signal, nstart, self.yn_size_p)
        #we have the pairs y_{p-1}, x_{p-1}, where x_{p-1} = 0 (transcient)
        
        # Computes the value of x_{p} from x_{p-1}, y_{p-1}, W_res, W_fb
        # Updated value of x_{p} is saved as self.xstate
        # After the update we have the state x_{p}, we will save states from here 
        self.update_state(yn) 
        
        num_saved = 0 # Counter to count number of saved states

        # States x are saved from index n=p to n=N-1, 
        # a total of N-p saved states
        num_states = self.N - self.yn_size_p

        # Initializee a numpy matrix to save states x_{n}
        # from n=p to n=N-1
        Xstates = np.zeros(shape=(num_states, self.num_neur))
        
        for n in range(self.yn_size_p, self.N):
            
            #we update yn to pair xn, to create a list of N-p values of y_{n} (for feedback)
            yn.appendleft(train_signal[n]) 
            
            #xn is already ready
            Xstates[num_saved, :] = self.xstate #save
            
            # counter updated
            num_saved += 1 
            
            # This updates the value of self.xstate using new value
            # of yn (feedback vector), and self.xstate
            self.update_state(yn)

        #after the loop we have the final state x_{N}, not saved in Xstates
        # So, self.Xstates only contains values from x_{p-1} to x_{N-1}.
        self.Xstates = Xstates #the function predict will need the Xstates
        
        return Xstates
    
    def train(self, train_signal):
        """ This function optimizes the output parameters of ESN viz. W_out, b.
        This is achieved by solving a ridge regression problem using the 
        predictions from the reservoir and actual outputs

        Args:
            train_signal ([numpy.ndarray]): time series used as training data
        
        Returns:
            train_mse: Mean squared error obtained during training (after fitting)
        """
        #tau, q and p have conditions that the user must respect
        self.check_tau_q_and_p() 

        # Builds a matrix of dimension N-tau-p x q s.t.
        # Y_{n+tau}^{q} = [y_{p+tau}, y_{p+tau+1}, ..., y_{N-1}], 
        # with each y_{i} of dimension q x 1, and there are in total 
        # N-tau-p such training examples
        YntauVects = self.buildYntauVectors() 

        # Number of samples to be considered for training, which is equal
        # to N-tau-p
        num_vects = YntauVects.shape[0]

        #NOTE: self.Xstates contains N-p number of values already computed, 
        # so we take only N-p-tau values from the larger set of reservoir 
        # states to create our training data
        XstatesTrain = self.Xstates[0:num_vects, :] 
        '''
        #at n=N-1, but we can train only until n=N-tau-1. In other words,
        #YntauVects and XstatesTrain must have the same num of vectors
        '''
        # XstatesTrain and YntauVects must have the same number of vectors
        # for training
        self.reg_model, train_mse = solve_ridge_regr(XstatesTrain, YntauVects, 
                                                                    self.beta)

        return train_mse
    
    def predict(self):
        """ This function performs prediction for Echo State Network (ESN)
        using the trained regression model

        Returns:
            ypredNtau ([numpy.ndarray]): No. of steps to predict
            pred_indexes ([numpy.ndarray]): array of indices for predicting
            values
        """
        ypredNtau = self.reg_model.predict([self.xstate])[0] #xstate is x_N
        ypredNtau = np.flip(ypredNtau) #we want indexes progress as time does
        
        # Create prediction indices
        index_last_pred = self.N + self.tau # = M+tau
        index_first_pred = index_last_pred - (self.yntau_size_q - 1) # = M+tau-(q-1)
        pred_indexes = np.arange(index_first_pred, index_last_pred + 1)
        
        return ypredNtau, pred_indexes #ypred_{N+tau} 
    
    '''
    #def predict(self, tau, yntau_size_q, beta_regul):
    def predict(self, train_signal):
        
        #tau, q and p have conditions that the user must respect
        self.check_tau_q_and_p() 
        YntauVects = self.buildYntauVectors()
        num_vects = YntauVects.shape[0]
        XstatesTrain = self.Xstates[0:num_vects, :] #because Xstates ends
        #at n=N-1 but we can train only until n=N-tau-1. In other words,
        #YntauVects and XstatesTrain must have the same num of vectors
        self.reg_model, train_mse = solve_ridge_regr(XstatesTrain, YntauVects, 
                                                                    self.beta)
        ypredNtau = self.reg_model.predict([self.xstate])[0] #xstate is x_N
        ypredNtau = np.flip(ypredNtau) #we want indexes progress as time does
        
        # Create prediction indices
        index_last_pred = len(train_signal) + self.tau # = M+tau
        index_first_pred = index_last_pred - (self.yntau_size_q - 1) # = M+tau-(q-1)
        pred_indexes = np.arange(index_first_pred, index_last_pred + 1)
        
        return ypredNtau, train_mse, pred_indexes #ypred_{N+tau} 
    '''

    def buildYntauVectors(self):
        """ Build a list of target vectors for regression

        Returns:
            YntauVects [(numpy.ndarray)]: Matrix of prediction values ((N-tau-p) x q)
            [num_samples x dimension of 'yntau']
        """
        nstart = self.yn_size_p - 1 #the pairs xn, yn started at p-1
        yntau = build_history_deque(self.train_signal, nstart + self.tau, 
                                    self.yntau_size_q)
        # n will go from p to N-tau-1, because we go tau in the future
        num_vects = self.N - self.tau - self.yn_size_p
        YntauVects = np.zeros((num_vects, self.yntau_size_q))
        num_saved = 0
        for n in range(self.yn_size_p, self.N - self.tau):
            yntau.appendleft(self.train_signal[n + self.tau])
            YntauVects[num_saved, :] = yntau
            num_saved += 1
        return YntauVects #y_{n+tau} vectors between n=p to n=N-tau-1 
    
    def update_state(self, yn):
        """ This function computes the value of the next reservoir state
        using the previous value of the reservoir state x_{n} and 
        the corresponding value of the reservoir output y_{n} by:

        x_{n+1} = tanh(W_res * x_{n} + W_fb * y_{n})

        #NOTE: u_{n} doesn't seem to be used here

        x_{n} is stored in self.xstate. This function rewrites the 
        value of self.xstate with x_{n+1}

        Args:
            yn ([numpy.array]): Feedback vector corresponding to the 
            current time index 
        
        Returns: 
            None (self.xstate gets overridden with new result)
        """
        self.xstate = np.dot(self.Wres, self.xstate) + np.dot(self.Wfb, yn)                                                           
        self.xstate = np.tanh(self.xstate)
        return
        
    def check_tau_q_and_p(self):
        """ This function checks whether the value of q and tau
        are same or not. Also checks if the value of p + tau is
        lower than the training signal length
        """
        if self.tau < self.yntau_size_q-1:
            print("q is too big for the given tau")
            sys.exit("")
        if self.yn_size_p + self.tau > self.N - 1:
            print("p + tau must be lower than the train signal length")
            sys.exit()
        return

    
def build_history_deque(training_signal, indexn, histo_size):
    """ Building the history deque ('yn')

    Args:
        training_signal ([numpy.ndarray]): time series as a training signal
        indexn ([int]): the index n
        histo_size ([int]): the length of feedback vector

    Returns:
        yn_deque ([deque]): the list containing the set of time-series values
                            as feedback vector    
    """
    yn = np.zeros(histo_size)
    for i in range(histo_size):
        if indexn-i < 0:
            print("encountered negative index when building yn")
            sys.exit()
        yn[i] = training_signal[indexn-i]
    yn_deque = deque(yn, maxlen=histo_size)
    return yn_deque

def build_reservoir(num_neur, conn_per_neur, spec_rad):
    Wres = np.zeros((num_neur, num_neur))
    for i in range(num_neur):
        random_columns = np.random.randint(0, num_neur, conn_per_neur)
        for j in random_columns:
            Wres[i, j] = np.random.normal(scale=1)
    Wres = change_spectral_radius(Wres, spec_rad)
    return Wres

def build_Wfb(num_neur, yn_size_p, spec_rad):
    Wfb = np.random.normal(scale=1, size=(num_neur, yn_size_p))
    U, S, VH = np.linalg.svd(Wfb) #S contains the sqrt of eigenvs of Wfb*WfbH 
    Wfb = Wfb * np.sqrt(spec_rad) / np.max(S)
    #now the max eigenv of Wfb*(Wfb)T is equal to spec_rad 
    return Wfb
    
def change_spectral_radius(Wres, new_radius):
    eigenvalues = np.linalg.eig(Wres)[0]
    max_absolute_eigen = np.max(np.absolute(eigenvalues))
    return Wres * new_radius/max_absolute_eigen

def solve_ridge_regr(Xstates, Ytargets, beta_regul):
    reg_model = Ridge(alpha=beta_regul)
    reg_model.fit(Xstates, Ytargets)
    Ypreds = reg_model.predict(Xstates)
    train_mse = mean_squared_error(Ytargets, Ypreds)
    return reg_model, train_mse
