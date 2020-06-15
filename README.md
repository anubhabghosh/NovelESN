# NovelESN
A novel Echo State Network (ESN) algorithm for predicting very complicated time series without   
the numerical instability issues of the standard one.

## main.py
It receives a txt file with a time series (training data) and it predicts the continuation at certain future instants of time.
The code is run via terminal:  
```
python main.py training_data.txt outputfile.txt tau history_q beta test_data.txt
```
* training_data.txt: Path to a txt file with a single column of values representing a time series of M scalars {y<sub>0</sub>, y<sub>1</sub>, ...y<sub>M-1</sub>}
* outputfile: Path to the output file where the prediction will be written.
* tau: (integer) Jump in the future. If M-1 is the time index of the last training data value y<sub>M-1</sub>, tau indicates that the last predicted value will be y<sub>M+tau</sub>.
* history_q (integer) Determines the amount of predicted values and the beginning of the prediction. The total number of predicted values will be q and they will go from instant M+tau-(q-1) to M+tau.
* beta (optional, default=1e-10): Regularization parameter for a ridge regression equation. See the file equations_doc.pdf for more info. 
* test_data.txt (optional, default=None): path to a txt file with test data values. The test data is used to know if the prediction is good or not. A perfect prediction should match the test data. If provided, the program will show a plot comparing the prediction and the test data.


## Installation
The package "echo_state_network" contains a single module, "novel.py". Inside this module there's a main class, "NovelEsn",  
which can be used by the user to create an ESN object.  
There's no installation. The user can simply download the package "echo_state_network" and import the class   
in a Python script with the command:  
```
from echo_state_network.novel import NovelEsn
```
