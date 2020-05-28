# Quantum Generative Adversial Networks


## Requirements
The following is required to run this example:
 * `python3.6` or newer
 * `scipy`
 * `numpy`
 * `sympy`
 * `climin`: Use `python3 -m pip install git+https://github.com/BRML/climin.git --user` to install


## Usage
Use
```bash
python3 run.py
```
for running the project with default parameters.
Use
```bash
python3 run.py -h
```
to see all configurable parameters.

To be able to run within reasonable time on a desktop (say 5 minutes),
you may want to use
```bash
python3 run.py -dis 1 -ds 10 -dr 10 -dg 10 -id 20 -ig 20 -pa -sf
```
This sets:
 - discriminator type to 1, which trains faster
 - discriminator amount of items in a dataset to 10 synthetic, 10 real
 - generator amount of items in a dataset to 10
 - discriminator max iterations in minimize function to 20
 - generator max iterations in minimize function to 20
 - printing of accuracies before and after each network update
 - plotting of test output figures

## Stolen work
We found the tutorial's work was stolen from [here](https://github.com/GiggleLiu/QuantumCircuitBornMachine/blob/master/notebooks/qcbm_gaussian.ipynb)