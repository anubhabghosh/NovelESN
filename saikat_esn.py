#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    NOVEL ESN
    A novel ESN algorithm for predicting signal continuations.
    The novel algorithm solves the numerical instability issues of the 
    standard one.
     

    Copyright (C) 2000  Saikat Chatterjee and Aleix Espuña Fontcuberta

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

class EsnNewMethod(object):
    def __init__(self, train_past_p, num_neur=1000, conn_per_neur=10,
                                                               spectr_rad=0.8):
        self.train_past_p = train_past_p
        self.num_neur = num_neur
        
        self.xstate = np.zeros(num_neur)
        self.Wfb = build_Wfb(num_neur, train_past_p, spectr_rad)
        self.Wres = build_reservoir(num_neur, conn_per_neur, spectr_rad)
    
    def teacher_forcing(self, train_signal):
        #we will save the training data for the predict function
        self.train_signal = train_signal  
        self.N = len(train_signal)
        
        nstart = self.train_past_p - 1 #min index to build the first yn
        yn = build_history_deque(train_signal, nstart, self.train_past_p)
        #we have the pairs y_{p-1}, x_{p-1}, where x_{p-1} = 0 (transcient)
        self.update_state(yn) 
        #after the update we have the state x_{p}, we will save from here 
        
        num_saved = 0
        N = len(train_signal)
        #we will save from n=p to n=N-1, a total of N-p saved states
        num_states = N - self.train_past_p
        Xstates = np.zeros(shape=(num_states, self.num_neur))
        for n in range(self.train_past_p, N):
            #xn is already ready
            yn.appendleft(train_signal[n]) #we update yn to pair xn
            Xstates[num_saved, :] = self.xstate #save
            num_saved += 1 
            
            self.update_state(yn)         
        #after the loop we have the final state x_{N}, not saved in Xstates
        
        self.Xstates = Xstates #the function predict will need the Xstates
        return Xstates
        
    def predict(self, tau, pred_past_q, beta_regul):
        check_tau_pred_past(tau, pred_past_q)
        YntauVects = self.buildYntauVectors(tau, pred_past_q)
        num_vects = YntauVects.shape[0]
        XstatesTrain = self.Xstates[0:num_vects, :] #because Xstates ends
        #at n=N-1 but we can train only until n=N-tau-1. In other words,
        #YntauVects and XstatesTrain must have the same num of vectors
        reg_model, train_mse = solve_ridge_regr(XstatesTrain, YntauVects, 
                                                                    beta_regul)
        ypredNtau = reg_model.predict([self.xstate])[0] #xstate is x_N
        return ypredNtau, train_mse, reg_model #ypred_{N+tau} 
    
    def buildYntauVectors(self, tau, pred_past_q):
        nstart = self.train_past_p - 1 #the pairs xn, yn started at p-1
        yntau = build_history_deque(self.train_signal, nstart+tau, pred_past_q)
        # n will go from p to N-tau-1, because we go tau in the future
        num_vects = self.N - tau - self.train_past_p
        YntauVects = np.zeros((num_vects, pred_past_q))
        num_saved = 0
        for n in range(self.train_past_p, self.N-tau):
            yntau.appendleft(self.train_signal[n+tau])
            YntauVects[num_saved, :] = yntau
            num_saved += 1
        return YntauVects #y_{n+tau} vectors between n=p to n=N-tau-1 
    
    def update_state(self, yn):
        self.xstate = np.dot(self.Wres, self.xstate) + np.dot(self.Wfb, yn)                                                           
        self.xstate = np.tanh(self.xstate)
        return
        
def check_tau_pred_past(tau, pred_past):
    if tau < pred_past-1:
        print("q is too big for the given tau")
        sys.exit("")
    return

def build_history_deque(training_signal, indexn, histo_size):
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

def build_Wfb(num_neur, num_train_past, spec_rad):
    Wfb = np.random.normal(scale=1, size=(num_neur, num_train_past))
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
