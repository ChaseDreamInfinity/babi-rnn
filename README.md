# RNN for bAbI tasks
This is an implementation of basic RNN model for bAbI tasks using Tensorflow with the reference of [keras example] (https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py) and Stanford cs224d assignment.

# Requirements
* Python 3.5
* Tensorflow
* [bAbI dataset](http://fb.ai/babi) should be downloaded to the same folder with babi_rnn.py

# Usage
To train and test all 20 bAbI 1k tasks using single mode:
```
python babi_rnn.py
```

# To-Do
* LSTM, GRU cell
* Joint training

# Results
These results are test error rate (%) with default configuration.

|   Task   |   babi-rnn (single)   |   		
|:--------:|:---------------------:|
|    1     |         51.6           |
|    2     |         72.2           |
|    3     |        79.9           |
|    4     |         27.9           |
|    5     |        38.0           |
|    6     |         49.7           |
|    7     |        22.3          |
|    8     |        26.0           |
|    9     |        38.6           |
|    10    |        55.2           |
|    11    |        31.1          |
|    12    |        26.8           |
|    13    |        7.9           |
|    14    |        70.7           |
|    15    |        72.1           |
|    16    |        49.1           |
|    17    |        49.5          |
|    18    |        9.3         |
|    19    |        90.1         |
|    20    |        7.8           |
| **Mean** |      **43.79**         |
