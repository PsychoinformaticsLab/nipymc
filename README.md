NiPyMC
------
This repository contains the supplemental material for Westfall, Nichols, & Yarkoni (2016).

## Table of Contents
- [Reproducing the analyses](#reproducing-the-analyses)
    + [Installing the NiPyMC Python package](#installing-the-nipymc-python-package)
    + [Preparing the HCP data](#preparing-the-hcp-data)
    + [The analysis notebooks](#the-analysis-notebooks)
- [Simulations](#simulations)
    + [SPM false positive rate](#spm-false-positive-rate)
    + [Test statistic inflation](#test-statistic-inflation)
- [Literature survey](#literature-survey)
- [Figures](#figures)

## Reproducing the analyses 

Some high-level notes here.

#### Installing the NiPyMC Python package

NiPyMC requires a working Python interpreter (either 2.7+ or 3+). We recommend installing Python and key numerical libraries using the [Anaconda Distribution](https://www.continuum.io/downloads), which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), NiPyMC itself can be installed in one line using pip:

    pip install git+https://github.com/tyarkoni/nipymc

You'll also need to install PyMC3 in order to fit most models. You can install PyMC3 from the command line as follows (for details, see the full [installation instructions](pip install git+https://github.com/pymc-devs/pymc3) on the PyMC3 repository:

    pip install git+https://github.com/pymc-devs/pymc3

Once both packages are installed, you should be ready to fit models with NiPyMC.

NiPyMC requires working versions of numpy, pandas, matplotlib, patsy, pymc3, and theano. Dependencies are listed in `requirements.txt`, and should all be installed by the NiPyMC installer; no further action should be required.

#### Preparing the HCP data

#### Running the models

## Simulations

#### SPM false positive rate

The code and results for this simulation are fully contained in [this notebook](https://github.com/tyarkoni/nipymc/tree/master/simulations/xsim_false_positive.ipynb).

#### Test statistic inflation

This is a very large simulation that we ran on the [Lonestar 5](https://www.tacc.utexas.edu/systems/lonestar) supercomputer at the [Texas Advanced Computer Center](https://www.tacc.utexas.edu/home). The [`/simulations` directory](https://github.com/tyarkoni/nipymc/tree/master/simulations) contains the scripts that we deployed.

## Literature survey

A Google Docs spreadsheet containing the study-level results of our literature survey can be found [here](https://www.google.com/url?q=https://docs.google.com/spreadsheets/d/1KUgrEyPpDsdY0GHKirfrJSuvkAyEzO7r9y32srOCkTg/edit?usp%3Dsharing&sa=D&ust=1474344243179000&usg=AFQjCNGbPjUM3jcmhmNpPD24qxMCv4XK9Q).

## Figures

Code to reproduce all the figures in our paper can be found [here](https://github.com/tyarkoni/nipymc/tree/master/figures). Code to reproduce the figures from the Appendix can be found [here](https://github.com/tyarkoni/nipymc/tree/master/simulations/xsim_figures.R).