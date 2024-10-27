# ML From Scratch

This repository contains simple implementations for various machine learning algorithms from scratch.
The purpose is to implement the key ideas using the simplest possible code (mostly for personal learning purposes).

## Assumptions
- Library effiency is not the top priority, learning is. Therefore, productionization is not the plan.
- Only use 3rd party libraries for visualization and testing (e.g. comparing gradients with Pytorch.)
- 

## Bootstrap environment
`./bootstrap.sh` will install all the necessary dependencies.

Then you can activate the installed virtual environment by `source .venv/bin/activate`.

## Repository Structure
- `autograd/`: Contains the hand-built autograd engine
- `test/`: Test cases for autograd and algorithms
- `notebooks/`: Contains some simple implementation and visualizations of various ML algorithms
  - `.py` files: Contains the implementations of the algorithms.
  - `.ipynb` files: Contains the notebooks for the algorithms.


## License
MIT
