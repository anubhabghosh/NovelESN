# NovelESN
A novel Echo State Network (ESN) algorithm for predicting very complicated time series without   
the numerical instability issues of the standard one.

## main.py
It receives a txt file with the training data and it returns a file with the prediction.
The code is run via terminal:  
```
python main.py training_data.txt outputfile.txt tau history_q beta test_data.txt
```
* training_data.txt: path to a txt file with a single column of values representing a time series of scalars {y_0, y_1, ...y_M-1}
* outputfile: name of the output file the program will generate. The prediction will be written in a column.
* tau: (integer) Jump in the future. If M-1 is the time index of the last training data value y_M-1, tau indicates that the prediction will end at index M+tau.
* history_q (integer) Determines the number of predicted values and the beginning of the prediction. The prediction will begin at index M+tau-(q-1). 


## Installation
The package "echo_state_network" contains a single module, "novel.py". Inside this module there's a main class, "NovelEsn",  
which can be used by the user to create an ESN object.  
There's no installation. The user can simply download the package "echo_state_network" and import the class   
in a Python script with the command:  
```
from echo_state_network.novel import NovelEsn
```
