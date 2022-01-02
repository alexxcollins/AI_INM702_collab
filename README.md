# AI_INM702_collab

Authors Suen Chi Hang and Alex Collins

### Task 2 code structure

All executable files are in the `src` folder. The parent class `GenerateData` is held in `regression.py`. The child classes `ColinearX` and `Outlier` are held in `colinearity.py` and `Outlier.py` respectively. 

simulation.py - build functions to investigate impact of factors on various mean and variance estimates: print_coef, simulation, change_factor
Outlier_analysis.py - mainly uses simulation.py to plot and analyse the impact of different outlier properties on various estimates
correlation_analysis.py - mainly uses simulation.py to plot and analyse the impact of correlation on various estimates

### Task 3 code structure

Executaable files are in `src` folder. The neural net class we use in notebook and to produce results is `NN` and is in `NN2.py`. We have also kept `NN.py` which is a "first draft" of the final `NN` class.

The code is run and parameter testing done in `NN2_fMNIST_classification.ipynb` in `nb` folder.

NN_testing.py - just for testing functionality of NN.py and kept for record purpose only



### Task 4 code structure

All code is in the notebook `CIFAR-10_Image_Classification.ipynb`. 

The `task4/data` folder is where the notebook downloads data by default.
