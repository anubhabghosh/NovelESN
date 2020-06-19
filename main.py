#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
import matplotlib.pyplot as plt
from parser import define_parser
from echo_state_network.novel import NovelEsn


parser = define_parser()
namespace = parser.parse_args()

#the compulsary args
train_file = namespace.training_file
output_file = namespace.output_file
tau = namespace.tau
yntau_size_q = namespace.history_q

#the optional args
yn_size_p = namespace.history_p
beta = namespace.beta
test_file = namespace.test_file

#loading the train and test data if any
train_signal = np.loadtxt(train_file)
if test_file:
    test_signal = np.loadtxt(test_file)

esn = NovelEsn(yn_size_p) #initialization of the esn
esn.teacher_forcing(train_signal) #feedback about the train data

prediction = esn.predict(tau, yntau_size_q, beta)[0] #pred of q values
#given a training time series y_0, y_1, y_2 ...y_M-1, the program will predict:
# y_M+tau-(q-1), y_M+tau-(q-2) ... y_M+tau
np.savetxt(fname=output_file, X=prediction)

index_last_pred = len(train_signal) + tau # = M+tau
index_first_pred = index_last_pred-(yntau_size_q-1) # = M+tau-(q-1)
pred_indexes = np.arange(index_first_pred, index_last_pred+1)

#Training data plot
plt.figure()
plt.title("Training data vs time index")
plt.plot(train_signal)

#Prediction plot
plt.figure()
plt.title("Prediction vs time index")
plt.plot(pred_indexes, prediction, label="prediction", color="green")
if test_file:
    plt.plot(pred_indexes, test_signal, label="test signal", color="orange")
plt.legend()

plt.show()
    
