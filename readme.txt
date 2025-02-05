This is the PyTorch code implementation and usage instructions for LiteMRINet.

1. Dependencies and Requirements

1). The code is implemented using the PyTorch framework, so PyTorch and torchvision need to be installed. The Python version we are using is 3.6.8, and the PyTorch version is 2.0.0+cu118.
2). Since we need to use ctypes, glob, time, os, tqdm, PIL, numpy, and timm, these libraries also need to be installed.

2.Description and Implementation of Key Algorithms

The implementation code for the algorithm is found in the LiteMRINet.py file. It mainly consists of multiple large-scale shared local feature extraction modules, a global feature extraction module using transformers, and a decoding module similar to UNet.

3.Training and Evaluation

The Train.py file is used for training and evaluating the model. The config.py file is used to set the corresponding parameters. LiteMRINet.py is the code implementation file for our proposed algorithm. The specific training steps are as follows:
1). Download the provided dataset, where the label values have been converted to 0 and 1.
2). Download the code and set the dirPath in the config file to the path of the dataset on your own computer.
3). Use the train.py file to start training. Each epoch will train on the training set and then test on the validation set, with the test metrics being saved in a txt file in the modename subdirectory under the dataset folder directory.