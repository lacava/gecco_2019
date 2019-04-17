Code to reproduce the experiments in the paper 
"Semantic variation operators for multidimensional genetic programming"

# Experiments

Experiments can be run using `analysis/ml-analyst/submit_jobs.py`. See the command line options (`python submit_jobs.py -h`) for help.

As example, this command would launch the entire experiment:

    python submit_jobs.py --r -ml FeatTuned,FeatSXOTuned,FeatRXOTuned,MLPmod,Linear,XGBoost -n_trials 10 -data ../penn-ml-benchmark/datasets/ -results results/

# Notebooks

`analysis/` contains these notebooks:
 - `results_tuning.ipynb` produces the tuning results figures.
 - `results_benchmark.ipynb` produces the comparisons to the [Where are we now? paper](https://github.com/EpistasisLab/regression-benchmark).
 - `stats.ipynb` depends on `results_benchmark.ipynb` and `results_tuning.ipynb', produces the statistical tests.
 - `results_benchmark-extended.ipynb` contains code to reproduce the extended PMLB results. 

# Dependencies
 
 - [scikit-learn](http://scikit-learn.org/stable/index.html)
 - [xgboost](https://github.com/dmlc/xgboost)
 - datasets come from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks)
 - [FEAT](https://github.com/lacava/feat)
