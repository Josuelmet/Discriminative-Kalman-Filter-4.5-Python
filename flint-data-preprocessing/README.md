## Preprocessing

We preprocess the Flint dataset in the following way:

1. Download data [here](https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012).
2. Run [flint_preprocess_data.ipynb](flint_preprocess_data.ipynb) to:
* move the data from the separate files and nested structs into a single location,
* and to coarsen data and hit the neural data with PCA and z-scoring.
* (After the preprocessing, many sets were only ~6000 pairs long; so each run used 5000 training points and 1000 test points.)
