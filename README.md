# NovelESN
A novel Echo State Network (ESN) algorithm for predicting very complicated time series without   
the numerical instability issues of the standard one.

## main.py
It receives a txt file with a time series (training data) and it predicts its continuation at certain future instants of time. The program will save and plot the prediction obtained.
The code is run via terminal:  
```
python main.py training_data.txt outputfile.txt tau history_q beta test_data.txt
```
* training_data.txt (string type): Path to a txt file with a single column of values representing a time series of M scalars {y<sub>0</sub>, y<sub>1</sub>, ...y<sub>M-1</sub>}.
* outputfile (string type): Path to the output file where the prediction will be written.
* tau (integer type): Jump in the future. If M-1 is the time index of the last training data value y<sub>M-1</sub>, tau indicates that the last predicted value will be y<sub>M+tau</sub>.
* history_q (integer type): Determines the amount of predicted values and the beginning of the prediction. The total number of predicted values will be q and they will go from instant M+tau-(q-1) to M+tau.
* beta (optional, float type, default=1e-10): Regularization parameter for a ridge regression equation. See the file equations_doc.pdf for more info. 
* test_data.txt (optional, string type, default=None): Path to a txt file with test data values. The test data is used to know if the prediction is good or not. A perfect prediction should match the test data. If provided, the program will show a plot comparing the prediction and the test data.
* history_p (optional, integer type, default=32): Number of time series values used to update the reservoir of neurons during training. See equations_doc.pdf for more info.
### algorithm overview
Given a time series of M scalars {y<sub>0</sub>, y<sub>1</sub>, ...y<sub>M-1</sub>}, the user will specify a jump in the future tau and a history q. The program will then return a set of predicted values {y<sub>M+tau-(q-1)</sub>, y<sub>M+tau-(q-2)</sub>, ...y<sub>M-tau</sub>}


