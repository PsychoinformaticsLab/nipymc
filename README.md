NiPyMC
------
This repository contains the supplemental material for Westfall, Nichols, & Yarkoni (2016).

## Table of Contents
- [Reproducing the analyses](#reproducing-the-analyses)
    + [Overview](#overview)
    + [Installing the NiPyMC Python package](#installing-the-nipymc-python-package)
    + [Downloading and preparing the HCP data](#downloading-and-preparing-the-hcp-data)
    + [Running the models](#running-the-models)
- [Simulations](#simulations)
    + [SPM false positive rate](#spm-false-positive-rate)
    + [Test statistic inflation](#test-statistic-inflation)
- [Literature survey](#literature-survey)
- [Figures](#figures)

## Reproducing the analyses 

#### Overview

Reproducing the analyses in the paper involves the following steps, which we will walk through in detail: 

1. **Installing the NiPyMC Python package from this Github page.** We wrote NiPyMC to serve as a high-level interface for fitting Bayesian mixed models to fMRI data using [PyMC3](https://pymc-devs.github.io/pymc3/index.html) in order to simplify the data analysis for our paper. We analyzed 5 of the 6 datasets presented in the paper using NiPyMC.
2. **Downloading and preparing the Human Connectome Project (HCP) datasets.** The data for the NeuroVault IAPS study and the OpenfMRI emotion regulation study are posted here in the [`data` directory](data). The other four datasets come from the HCP. These datasets are far too large to post here (and this would also be contrary to the HCP's [data use terms](http://www.humanconnectome.org/data/data-use-terms/)), so you will need to download the HCP data separately and use [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) to extract the data from the 100 regions of interest (ROIs) that we use in our analyses.
3. **Running the models.** This part is mostly straightforward, but fitting the full set of models to all of the datasets is extremely computationally demanding. We ran our analyses on the [Lonestar 5](https://www.tacc.utexas.edu/systems/lonestar) supercomputer at the [Texas Advanced Computing Center](https://www.tacc.utexas.edu/home), and if you want to re-estimate all the models yourself, you will want to use some computing cluster. For convenience we have posted compressed summaries of all of the fitted models from our analyses (the Python dictionaries returned by [`BayesianModelResults.summarize()`](nipymc/model.py)) in the [`results`](results) directory, which you can use, for example, to reproduce the figures and explore the parameter estimates for individual models.

#### Installing the NiPyMC Python package

NiPyMC requires a working Python interpreter (either 2.7+ or 3+). We recommend installing Python and key numerical libraries using the [Anaconda Distribution](https://www.continuum.io/downloads), which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), NiPyMC itself can be installed in one line using pip:

    pip install git+https://github.com/PsychoinformaticsLab/nipymc

You'll also need to install PyMC3. You can install PyMC3 from the command line as follows (for details, see the full [installation instructions](pip install git+https://github.com/pymc-devs/pymc3) on the PyMC3 repository:

    pip install git+https://github.com/pymc-devs/pymc3

Once both packages are installed, you should be ready to fit models with NiPyMC.

NiPyMC requires working versions of numpy, pandas, scipy, scikit-learn, pymc3, and theano. Dependencies are listed in `requirements.txt`, and should all be installed by the NiPyMC installer; no further action should be required.

#### Downloading and preparing the HCP data

You can download the HCP data [here](http://www.humanconnectome.org/data/).

Once you have the data, things will work smoothest if you put it in the following subdirectories in your `nipymc` directory:

- `data/eprime`: This will contain all of the E-prime experiment files for all subjects for all tasks. It will look like a bunch of text files with names like `EMOTION_100307_RL_TAB.txt`.
- `data/onsets`: This will contain the text files with information about when the different stimulus categories were presented to each subject in each task. It will look like a bunch of text files with names like `EMOTION_100307_RL_fear.txt`.
- `HCP/data`: This will contain the actual fMRI time series data. There will be a separate subdirectory for each subject (e.g., a directory labeled `100307` for subject 100307).

Now you can use [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) to extract the time series for the 100 ROIs we used in our analyses and save this extracted ROI data to the `data` directory. You can do this from the command line using the `fslmeants` command, like so:
```
fslmeants -i [img_file] --label=[roi_file] -o [output_file]
```
Replace `[img-file]` with the raw time series data file from which you want to extract the ROI data (without the brackets), `[roi_file]` with the file containing info about how the ROIs are defined, and `[out_file]` with the location and filename where you want to save the resulting ROI time series data.

For example, to extract the 100 ROI time series for run #1 (out of 2 runs total) for subject # 100307 in the HCP Emotion task, we make the following substitions in the `fslmeants` command above:

[img_file] = 
`HCP/data/100307/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_RL.nii.gz`

[roi_file] =
`data/masks/whole_brain_cluster_labels_PCA=100_k=100.nii.gz`

[output_file] =
`data/EMOTION_timeseries/100307_1.txt`

The most efficient thing to do is to write a shell script or Python script to do this for all tasks and subjects of interest.

#### Running the models

With the ROI time series for the HCP datasets all neatly extracted, estimating the models is simply a matter of running the scripts in the [`analyses` directory](analyses). For each task and model, running the shell script will call the associated Python script 100 times, which fits the model to each of the 100 ROIs. Because of the computational demands, fitting the models for all 100 ROIs pretty much requires a computing cluster of some kind. For the analyses in this paper, fitting a single model takes anywhere from a few minutes to several hours, depending on the specific dataset and model.

All of the shell and python scripts in the `analyses` directory are posted exactly as we ran them on [Lonestar 5](https://www.tacc.utexas.edu/systems/lonestar). However, you will probably want to change the directories that are referenced in the shell and Python scripts before re-running yourself.

As we wrote above, for convenience we have posted compressed summaries of all of the fitted models from our analyses (the Python dictionaries returned by [`BayesianModelResults.summarize()`](nipymc/model.py)) in the [`results`](results) directory, which you can use, for example, to reproduce the figures and explore the parameter estimates for individual models.

## Simulations

#### SPM false positive rate

The code and results for this simulation are fully contained in [this notebook](simulations/xsim_false_positive.ipynb).

#### Test statistic inflation

This is a very large simulation that we ran on the [Lonestar 5](https://www.tacc.utexas.edu/systems/lonestar) supercomputer at the [Texas Advanced Computing Center](https://www.tacc.utexas.edu/home). The [`simulations` directory](simulations) contains the scripts that we deployed.

## Literature survey

A Google Docs spreadsheet containing the study-level results of our literature survey can be found [here](https://www.google.com/url?q=https://docs.google.com/spreadsheets/d/1KUgrEyPpDsdY0GHKirfrJSuvkAyEzO7r9y32srOCkTg/edit?usp%3Dsharing&sa=D&ust=1474344243179000&usg=AFQjCNGbPjUM3jcmhmNpPD24qxMCv4XK9Q).

## Figures

Code to reproduce all the figures in our paper can be found [here](figures). Code to reproduce the figures from the Appendix can be found [here](simulations/xsim_figures.R).

Note that reproducing Figures 2 through 6 requires the full fitted model objects that contain the individual MCMC samples for four of the fitted models. These fitted model objects are too large to host in this Github repository (they range from 112MB to 242MB), so we have uploaded them to a [secondary OSF repository](https://osf.io/84yq2/files/). You can download the objects there and then place them in the [`results/emo_results`](results/emo_results) and [`results/lang_results`](results/lang_results) directories.
