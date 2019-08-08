[![DOI](https://zenodo.org/badge/DOI/)](https://doi.org)

CycleJet
========

This repository contains the code and results presented in [arXiv:1908.xxxxx](https://arxiv.org/abs/1908.xxxxxx).

## About

CycleJet is a framework to create mappings between different categories of jets using CycleGANs.
The model architecture is adapted from https://github.com/eriklindernoren/Keras-GAN/tree/master/cyclegan.

## Install CycleJet

CycleJet is tested and supported on 64-bit systems running Linux.

Install CycleJet with Python's pip package manager:
```
git clone https://github.com/JetsGame/CycleJet.git
cd CycleJet
pip install .
```
To install the package in a specific location, use the "--target=PREFIX_PATH" flag.

This process will copy the `cyclejet` program to your environment python path.

We recommend the installation of the CycleJet package using a `miniconda3`
environment with the
[configuration specified here](https://github.com/JetsGame/CycleJet/blob/master/environment.yml).

CycleJet requires the following packages:
- python3
- numpy
- [fastjet](http://fastjet.fr/) (compiled with --enable-pyext)
- matplotlib
- pandas
- keras
- tensorflow
- json
- gzip
- argparse
- scikit-image
- scikit-learn
- hyperopt (optional)


## Pre-trained models

The final models presented in
[arXiv:1908.xxxxx](https://arxiv.org/abs/1908.xxxxx "gLund paper")
are stored in:
- results/QCD_to_W: CycleJet which converts QCD <-> W jet.
- results/parton_to_delphes: CycleJet which converts partons <-> delphes.

## Input data

All data used for the final models can be downloaded from the git-lfs repository
at https://github.com/JetsGame/data.

## Running the code

In order to launch the code run:
```
cyclejet --output <output_folder>  <runcard.yaml>
```
This will create a folder containing the result of the fit.

## References

* S. Carrazza and F. A. Dreyer, "Towards a generative model for jet substructure,"
  [arXiv:1908.xxxxx](https://arxiv.org/abs/1908.xxxxx "gLund paper")
