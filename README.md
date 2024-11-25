# Source code for simulations of arxiv::240401477

Here, we present the source code for simulations from our pre-print ["Building a fusion-based quantum computer using teleprted gates"](https//::arxiv.org/abs/2404.01477).
We investigated properties of the introduced fusion networks that are based on using $4$-qubit linear cluster and $10$-qubit branch-like cluster as resource states.
In the simulations we estimate the probability of succesfull decoding for a given values of erasures and error probabilitites.

The source code is contained in `deconding_simulation.py`. It uses [PyMatching 2](https://github.com/oscarhiggott/PyMatching) library to perform simulations. The source of PyMatching library is contained in `./PyMatching` folder. We added some new functionality to original Pymatching library in order to deal with erasures, see functions `percolate_batch` and `decode_batch_with_erasures`. The first one is used for detecting percolation of erasures for the batch of samples. The second one performs decoding with accounted erasures. It also does it for the batch of samples. The C++ implementation can be found in `./PyMatching/src/pymatching/sparse_blossom/driver/user_graph.pybind.cc` and `./PyMatching/src/pymatching/sparse_blossom/driver/user_graph.cc`.

Before installing PyMatching from source make sure that there is no folder `./PyMatching/src/pymatchin/__pycache__`. If it exists, then delete:

- `rm -r ./PyMatching/src/pymatching/__pycache__`

To install PyMatching simply run this command from the folder containing `decoding_simulation.py` and `./PyMatching`:

- `pip install ./PyMatching`

To run sumulations type the command of the form:

- `python3 decoding_simulation.py id S S S M [M ...] C_err C_ers R R R N`

Here, `id` is integer number corresponding to selected for simulation syndrome graph. There are 5 possible variants:
- `id=0` - toric code syndrome graph
- `id=1` - syndrome graph for 'star' model from https::/arxiv.org/abs/2101.09310
- `id=2` - syndrome graph for the four-qubit fusion networks described in our paper
- `id=3` - syndrome graph for the ten-qubit fusion networks described in our paper
- `id=4` - syndrome graph for one experimental model

Parameters `S S S` correspond to linear sizes of the simulated syndrome graphs.
Parameters `M [M ...]` denote the membranes ids. In short, for 3D syndrome graph one need to set them as `x y z` and for 2D syndrome graph as `x y`.
`C_err` and `C_ers` denote error and erasure coefficients. For some parameter `x` probability of erasure is equal to `C_ers * x` and probability of error is equal to `C_err * x`.
`R R R` set the range of the parameter `x` in the format `[start, end, step]`.
`N` is the number of samples for every value of erasure and error probabilis.
If you want to track time of operations, use the flag `--debug`.

Let us present some examples:

- `python3 deconding_simulation.py 2 12 16 20 x y z 0.0 1.0 0.01 0.10 0.01 10000` - this command run the process of simulation of syndrome graph of the four-qubit fusion network for the case of zero-rate errors. The range of erasure probabilities: 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09. The syndrome graphs of sizes 12, 16 and 20 are used.
- `python3 deconding_simulation.py 2 12 16 20 x y z 1.0 0.0 0.001 0.010 0.001 10000` - this command run the process of simulation of syndrome graph of the four-qubit fusion network for the case of zero-rate erasures. The range of erasure probabilities: 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009. The syndrome grapha of sizes 12, 16 and 20 are used.
- `python3 deconding_simulation.py 2 12 16 20 x y z 0.00464 0.0128 0.99 1.02 0.01 10000` - this command run the process of simulation of syndrome graph of the four-qubit fusion network. The range of erasure probabilities: 0.99 * 0.0128, 1.0 * 0.0128, 1.01 * 0.0128. The range of error probabilities: 0.99 * 0.00464, 1.0 * 0.00464, 1.01 * 0.00464. The syndrome graphs of sizes 12, 16 and 20 are used.

The results of simulations are stored in `results` folder. The data used in our paper can be found in `paper_data`.
