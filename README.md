# NovelESN
A novel Echo State Network (ESN) algorithm for predicting very complicated time series without   
the numerical instability issues of the standard one.

## main.py
It receives a txt file with the training data and it saves and plots the prediction obtained.
The code is run via terminal:  
```
python main.py training_data.txt outputfile.txt tau history_q beta test_data.txt
```
* training_data.txt: Path to a txt file with a single column of values representing a time series of M scalars {y_0, y_1, ...y_M-1}
* outputfile: Path to the output file the program will generate. The prediction will be written in a column.
* tau: (integer) Jump in the future. If M-1 is the time index of the last training data value y_M-1, tau indicates that the last predicted value will be y_M+tau.
* history_q (integer) Determines the number of predicted values and the beginning of the prediction. The total number of predicted values will be q and they will go from y_{M+tau-(q-1) to y_{M+tau}}.
* beta (optional, default=1e-10): Regularization parameter for a ridge regression equation. See the file equations_doc.pdf for more info. 
* test_data.txt (optional, default=None): path to a txt file with test data values. The closer the prediction matches the test data, the better. If the test data file is provided, the program will show a plot comparing the prediction and the test data.


## Installation
The package "echo_state_network" contains a single module, "novel.py". Inside this module there's a main class, "NovelEsn",  
which can be used by the user to create an ESN object.  
There's no installation. The user can simply download the package "echo_state_network" and import the class   
in a Python script with the command:  
```
from echo_state_network.novel import NovelEsn
```
