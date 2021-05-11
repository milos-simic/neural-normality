# Testing for Normaity with Neural Networks

This is the code and data repository for the paper ["Testing for Normality with Neural Networks"](https://arxiv.org/abs/2009.13831).

## Requirements

In order to run the experiments, you should have `python` (`3.9`) and `jupyter` installed, as well as the following `python` packages:
* `numpy`
* `pandas`
* `scipy`
* `rpy2`
* `matplotlib`
* `dill`
* `kgof` (available [here](https://github.com/wittawatj/kernel-gof/tree/master/kgof))

Additionally, install `R` (`4.0.4`) and its libraries:
* `nortest`
* `gsl`
* `PearsonDS`. 
* `ggplot2`
* `ggpubr`
* `dplyr`
* `stringr`
* `unglue`
* `caret`
* `latex2exp`

## How to replicate the experiment

1. Run `configure.ipynb`
2. (optional) If you want to recreate the data, run `generate_sets.ipynb`. If you want to use the datasets that were used in the original experiment, skip this step, but extract the csv files from the zip files and make sure that they are located in the data directory.
3. (optional) If you want to train the SBNN classifier, run `sbnn.ipynb`. If you want to use the already trained SBNN, skip this step.
4. Run `descriptor_based_neural_network.ipynb`. This is the main notebook. It trains and evaluates neural networks for normality testing. Just follow the instructions there.
5. (optional) If you want to recreate the plots that visualize the results of cross-validation, run notebook `analyze_cross_validation.ipynb`. As it is, the notebook contains the most recent plots.
6. (optional) If you want to run simulations to estimate the critical values for each `n=1, 2, ..., 99, 100` and `alpha=0.01, 0.02, ..., 0.98, 0.99`, run notebook `derive_the_table_of_critical_values.ipynb`. You can analyze the critical values if you run `analyze_critical_values.ipynb`.
7. (optional) If you want to find the critical values for the robustified tests by yourself, run `make_robust_tests.ipynb`. If not, you can use the critical values found in the original experiment, just make sure that you have file `robust_critical_values.p` and that you can `unpickle` it.
8. (optional) If you want to find the optimal decision thresholds for DBNN by yourself, run `find_optimal_thresholds.ipynb`. Otherwise, you can use the threshold I found in the original experiment. Make sure that you have file `optimal_thresholds.p` and that you can `unpickle` it.
9. To compare DBNNs to selected classical and robustified normality tests, as well se SBNN and a kernel test of goodness-of-fit, run the following notebooks:
    9.1. `run_C.ipynb` (power analysis),
    9.2. `run_D.ipynb` (overall performance analysis),
    9.3. `run_F.ipynb` (robustness analysis), and
    9.4. `run_R.ipynb` (comparison using real-world data).
