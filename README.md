Kaggle Competitions
====
This repository will present my solutions to various Kaggle competitions using PyTorch framework. Detailed explanations will be availble at [xu932.github.io](https://xu932.github.io)

Organization
----
All solutions are organized in under directories: `model` and `src`. The `model` folder contains subfolders named by the computition. Data used for the competition can be downloaded from [Kaggle](https://www.kaggle.com/).

#### `model/<compition>` folder contains:
* `model.pt`: the binary file that contains the state of the trained model
* `accu.eps`: the graph that illustrates the accuracy of the model on validation set in terms of number of epochs trained
* `loss.eps`: the graph that illustrates the loss of the model on validation set in terms of number of epochs trained

#### `src` folder contains:
* `<competiton>.py`: the source code that used to the competition. _The code uses GPU for both training the testing; modifications may be required to run on machines that does not have GPU.
* `common.py`: used to contain common functionalities for training/testing various models, such as training procedure and testing procedure
