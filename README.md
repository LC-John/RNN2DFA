# RNN to DFA

This is a project aiming to extract a minimal DFA from a well-trained RNN model.

## Tomita Grammars

The Tomita grammars are a set of widely used benchmark grammars in the problem of grammar inference. It contains 7 different regular grammars defined upon the binary alphabet <img src="http://latex.codecogs.com/gif.latex?\Sigma=\{0,1\}" />, as shown below.

```
Tomita 1    1*
Tomita 2    (10)*
Tomita 3    all strings without containing odd number of consecutive 0's after odd number of consecutive 1's
Tomita 4    all strings without containing 3 consecutive 0's (000)
Tomita 5    all strings with even numbers of 0's and 1's
Tomita 6    all strings satisfying #(0)-#(1) = 3n (n=...,-1,0,1,...)
Tomita 7    1*0*1*0*
```

The corresponding minimal DFA's are shown below. The number of states are less than 5. States with thick border represent ACC states, while states with thin border represent REJ states. States with an "S" are starting staes of each DFA. The DFA's are equivalent to Tomita 1 to Tomita 7, from left to right.

![tomita grammars dfa](./images/TomitaDFA.jpg)

The 7 DFA's are defined in ```./tomita/tomita.py```, which are able to classify a given sequence (ACC/REJ) and generate sequences with their corresponding ACC/REJ labels. As the 7 datasets are large to some degree, and they are able to be automatically generated, they are excluded from this repo. **REMEMBER TO GENERATE THE DATASET BEFORE TRAINING THE RNN MODELS.** Run ```python3 generator.py``` under the directory ```./tomita/``` to generate the 7 datasets of the Tomita grammars. 