# pyODPlus
pyODPlus is an extension of [PyOD](https://github.com/yzhao062/pyod/tree/master/pyod/models), a Python toolkit for detecting outlying objects in multivariate data. PyOD includes more than 30 detection algorithms from classical LOF (SIGMOD 2000) to the latest COPOD (ICDM 2020). 

## Getting Started
A `requirements.txt` file is provided in the main directory and contains all packages that have to be installed. To install all the packages, run the following code section:
```
pip install -r requirements.txt
```

## Implemented Algorithms
Implementations of the following algorithms can be found in the `outlier_detection` directory. Demonstrations of how to load and use these algorithms are also available in the `demo` directory. 

### Relative Outlier Cluster Factor (ROCF)
ROCF is an outlier detection method proposed by Huang, et. al. in ["A novel outlier cluster detection algorithm without top-n parameter"](https://doi.org/10.1016/j.knosys.2017.01.013), published in Elsevier Knowledge-Based Systems 121 (2017) pp.32-40. The ROCF algorithm aims eliminate the need to specify the number or percentage of outliers in the dataset, i.e. n parameter or the contamination parameter in PyOD.
```
rocf = ROCF() 
rocf.fit(X)
rocf.get_outliers()
```
### Cluster Based Outlier Factor (CBOF)
CBOF is an outlier detection method proposed by Duan, et. al. in ["Cluster-Based Outlier Detection"](https://www.researchgate.net/publication/261018177_Cluster_based_Outlier_Detection), published in Ann. Oper. Res. 168 (1) (2009) pp.151–168. The paper introduces the clustering-based approach to detect not just single point outliers (noise), but also small clusters of outliers. 
```
cbof = CBOF() # create instance of model
cbof.fit(X) # fit model on X
cbof.get_outliers() # retrieve outliers
```

## Evaluation
The `evaluation` directory contains Jupyter Notebooks with code used to evaluate the performance of the ROCF model on several datasets. We tested the ROCF algorithm on 3 types of datasets, namely:
1. Resampled IRIS (replicated from paper)
2. Synthetic D1, D2 and D3 (replicated from paper)
3. Bank Card Fraud Transactions (https://www.kaggle.com/ninads/kernel3b5cdd2865/data) 

The `utils` directory includes code to reproduce the same synthetic datasets used for our evaluation. 
