#### NOTE: This is a forked version of the repository to compare NovelESN with other recurrent architectures like AR model, LSTM and GRU.

## Details regarding data

- Data description: `data/dynamo_esn.txt` represents solar cycle data that is **synthetically generated** through simulation (solving a differential equation). `data/solar_data.txt` contains the **real** data corresponding to solar cycles. 
  - Both the datasets have time indices that represent *months*. So the data is *number of sunspots* vs. *months*
  - One **solar cycle** approximately corresponds to 11 years. A **cycle** is considered as a set of points that go from a minimum to the next maximum (*TODO: function for extracting data corresponding to a single cycle*)
  - The **real** solar cycle data has been pre-processed *a priori* by a moving-average filter consisting of 13 taps.
  - There is also a *toy* dataset consisting of a clean, sinusoidal signal f(x) = sin(x), which can be used as sanity check for new algorithms.

- Data preparation: 
  - Having the entire `dynamo` data in a single file (`data/dynamo_esn.txt`), a specific cycle was chosen. All the data before the beginning of the cycle was put as the training data. The data corresponding to the specific chosen cycle was the test data. 
  - Usually `dynamo_esn.txt` consists of data in two columns. The first column usually corresponds to time index (useful for plotting), and the second column corresponds to the actual value of the time-series. 
  - It is essential to normalize between [0, 1] the signals that are sent to the algorithm. Otherwise the tanh function can saturate because the recursive equation involved could become unstable. Normalization between 0 and 1 is not the only one that can work, maybe also like [-1,1], but it is assumed [0, 1] is better.
 
- TODOs:
  - [ ] Function for extracting data corresponding to a single cycle or cycles. Also split data into train, validation and test if possible.

## NovelESN (Novel Echo State Network)
- Key features of the NovelESN model as compared to a standard ESN:
  - `history_p`: This parameter (p) decides the dimensionality of the feedback vector in the update equation for reservoir states. The value of p was chosen using some validation set, possibly the test set (alothough in the paper, it is mentioned trial and error). The optimal predictions were found for values of p in the range of something like 16-32, and value of p=32 was chosen. *NOTE: This could be fixed better by grid search or some other technique*
  - `history_q`: Predicition of several future values instead of a single value
  - `tau`: 'Jump' in future for prediction
  - `beta`: Regularization parameter for Ridge regression.
- Other mathematical details found in [`equations_doc.pdf`](https://github.com/anubhabghosh/NovelESN/blob/master/NovelESN_Aleix/equations_doc.pdf)

- TODOs:
  - [ ] Restructuring of code to make it easier to read and more flexible to arguments. So that comparisons can be easier with other methods.

## Linear Auto-Regressive model
### Trying to run grid-search for Linear AR model using `main_gs.py`
For example, if it is required to run main_gs.py for predicting the cycle for cycle index 75
(cycle 76 - future cycle for *dynamo*, and cycle 23 - future cycle for *solar*)
```
python main_gs.py --model_type linear_ar --dataset dynamo --train_file ./data/dynamo_esn.txt --output_file models/dynamo_pred_linear_ar_75.txt --predict_cycle_num 75 --grid_search 1
```
## Long-Short Term Memory
### Trying to run LSTM on `main_gs.py`
For example, if it is required to run main_gs.py for predicting the cycle for cycle index 75
(cycle 76 - future cycle for *dynamo*, and cycle 23 - future cycle for *solar*)
```
python main_gs.py --model_type lstm --dataset dynamo --train_file ./data/dynamo_esn.txt --output_file models/dynamo_pred_lstm_75.txt --predict_cycle_num 75 --grid_search 0
```
## Gated Recurrent Units
### Trying to run GRU on `main_gs.py`
For example, if it is required to run main_gs.py for predicting the cycle for cycle index 75
(cycle 76 - future cycle for *dynamo*, and cycle 23 - future cycle for *solar*)
```
python main_gs.py --model_type gru --dataset dynamo --train_file ./data/dynamo_esn.txt --output_file models/dynamo_pred_gru_75.txt --predict_cycle_num 75 --grid_search 0
```

