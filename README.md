# NovelESN
A novel Echo State Network (ESN) algorithm for predicting very complicated time series without   
the numerical instability issues of the standard one.

## What the program does
A standard ESN is a neural network capable to receive a time series (training) and predict its inminent continuation for a short time. We are presenting here a new algorithm. This new method can directly predict the signal in any desired future instants of time, without having to predict the inminent continuation first. Just to give the reader an example, imagine we use a training signal from time t=1sec to t=100sec. We can then ask the program to make a jump of 20sec in the future and return a history of 3 values. That command will directly return 3 predicted values at t=118sec, t=119sec and t=120sec. The previous values from t=111sec to t=117sec will not have been calculated. 

## Installation
The package "echo_state_network" contains a single module, "novel.py". Inside this module there's a main class, "NovelEsn",  
which can be used by the user to create an ESN object.  
There's no instalation. The user can simply download the package "echo_state_network" and import the class   
in a Python script with the command:  
```
from echo_state_network.novel import NovelEsn
```
