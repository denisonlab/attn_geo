# attn_geo

Data and analysis code for "Attention reshapes the representational geometry of a perceptual feature space" by Angus Chapman, Melissa Allouche and Rachel Denison.

The full manuscript is available as a preprint on [bioRxiv](https://doi.org/10.1101/2025.08.28.672962).

contact: Angus Chapman, angusc@bu.edu

## File summary

Home directory contains .m files to reproduce the model fitting and analyses presented in the manuscript.

- attn_geo_summary.m: example analysis code used to reproduce the main findings in the manuscript
- fit_triad_alt_models.m: fits the angular distance and RDM models
- fit_triad_cond.m: fits the geometric models that include separate coordinates for each attention condition
- fit_triad_nocond.m: fits geometric models aggregated across attention conditions

NB: fit_triad_[cond/nocond].m contain code necessary to run these scripts on the BU Shared Computing Cluster, and will require adjustments to work locally. Particularly lines 7, 14, and 53, which pull environmental variables used for parallel processing with different model parameters.

fit_triad_cond.m also relies on the unconstrained model fits from Step 1 in fit_triad_nocond.m for initial parameters (saved in ./data/fit_nocond/)

- parforTracker.m: helper function that tracks progress of model fitting within parfor loops
- pred_triad_angdist.m: takes in data and model parameters and returns NLL of the angular distance model
- pred_triad_distmat.m: takes in data and model parameters and returns NLL (and gradients) of the RDM model
- pred_triad_resp.m: takes in data and model parameters and returns NLL (and gradients) of the geometric model

---

We also include the following files in the data directory:

- fit_alt_models.mat: output of fit_triad_alt_models.m
- fit_geo_models.mat: output of fit_triad_cond.m and fit_triad_nocond.m
- triad_data_collated.mat: trial-level data for each participant, containing information about experimental conditions as well as behavioral responses to the attention task and triad similarity judgements.
