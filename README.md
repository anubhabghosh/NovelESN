# NovelESN
A novel Echo State Network (ESN) algorithm for predicting very complicated signal continuations without   
the numerical instability issues of the standard one.

## What the program does
An ESN is a neural network that behaves like a dynamical system. We can feed the network with a signal for some time.  
Once we stop, the neural network will try to predict its continuation.

## Installation
The package "echo_state_network" contains a single module, "novel.py". Inside this module there's a main class, "NovelEsn",  
which can be used by the user to create an ESN object.  
There's no instalation. The user can simply download the package "echo_state_network" and import the class   
in a Python script with the command:  
```
from echo_state_network.novel import NovelEsn
```
