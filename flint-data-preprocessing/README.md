## Preprocessing

We preprocess the Flint dataset in the following way:
(we have also uploaded the processed outputs [procd_velocities.npy](procd_velocities.npy) and [procd_spikes.npy](procd_spikes.npy) directly to the this folder, so you can skip downloading + preprocessing data entirely if you want):

1. Download data [here](https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012). Place the five .mat files in this directory.
2. Run [flint_preprocess_data.ipynb](flint_preprocess_data.ipynb) to:
* move the data from the separate files and nested structs into a single location,
* and to coarsen data and hit the neural data with PCA and z-scoring.
* (After the preprocessing, many sets were only ~6000 pairs long; so each run used 5000 training points and 1000 test points.)
* (After preprocessing, [procd_velocities.npy](procd_velocities.npy) and [procd_spikes.npy](procd_spikes.npy) should exist in this directory.)
