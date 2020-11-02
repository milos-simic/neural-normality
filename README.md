# Testing for Normaity with Neural Networks

This is the code and data repository for the paper ["Testing for Normality with Neural Networks"](https://arxiv.org/abs/2009.13831).

## Requirements

In order to run the experiments, you should have `python` (`3.6`, but higher versions should also work) and `jupyter` installed, as well as the following `python` packages:
* `numpy`
* `pandas`
* `scipy`
* `rpy2`
* `matplotlib`

Additionally, install `R` and its libraries `nortest`, `gsl`, and `PearsonDS`. In the original experiment, version `3.4.4` of `R` was used.

## How to replicate the experiment

1. Run `configure.ipynb`
2. (optional) If you want to recreate the data, run `generate_sets.ipynb`. If you want to use the datasets that were used in the original experiment, skip this step, but extract the csv files from the zip files and make sure that they are located in the data directory.
3. (optional) If you want to train the SBNN classifier, run `sbnn.ipynb`. If you want to use the already trained SBNN, skip this step.
4. Run `descriptor_based_neural_network.ipynb`. This is the main notebook. It conducts all the parts of the origina experiment, makes all the plots and prints out LaTeX reports. Just follow the instructions in the notebook. 
