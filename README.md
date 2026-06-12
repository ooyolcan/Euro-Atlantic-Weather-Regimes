# Identifying Atmospheric Regimes Driving Compound Renewable and Hydro-meteorological Anomalies in Europe

This repository contains the standardized Python codebase and derived datasets necessary to reproduce the atmospheric regime identification, compound anomaly analysis, and statistical significance testing presented in the associated manuscript.

## 1. Data Availability & Copernicus Licensing
The primary high-resolution atmospheric fields used in this study are derived from the ECMWF ERA5 reanalysis dataset. 
* **Data Retrieval:** Raw daily data for 100-meter wind components (`u100`, `v100`), surface solar radiation downwards (`ssrd`), total precipitation (`tp`), and 500 hPa geopotential height (`z500`) can be freely downloaded from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/).
* **Licensing:** All ERA5 data usage strictly adheres to the Copernicus licensing terms. The derived datasets are shared under a CC-BY 4.0 license, and the analysis scripts are provided under the MIT License to ensure full computational reproducibility.

## 2. Repository Structure
* `main_pipeline.py`: The core script that outlines the general methodological framework used for preprocessing, spatial clustering, and rigorous statistical validation.
* `joint_probability_and_bootstrap.py`: The script responsible for bivariate 2D Kernel Density Estimation (KDE) and the 1000-iteration bootstrapping procedure to calculate 95% Confidence Intervals for joint deficit probabilities.
* `regime_results.npz`: The derived cluster assignments containing the chronological regime labels. This is provided for reviewers and users lacking access to large computing resources to process the raw `.nc` files.

## 3. Reproduction Instructions

### A. Preprocessing and Anomaly Calculation
Run `main_pipeline.py`. The script applies a 5-day cyclic moving average to smooth the climatological baseline and computes daily deviations, preserving the physical units for surface fields. 

### B. Atmospheric Regime Identification (K-Means)
The spatial clustering is integrated within `main_pipeline.py`. It executes K-Means clustering directly on the full physical state space of the `z500` anomalies. Prior to clustering, a spatial area-correction is applied by scaling the grid points with the square root of the cosine of their latitude.

### C. Statistical Significance Testing
The downstream section of `main_pipeline.py` conducts the non-parametric Wilcoxon Signed-Rank Test to evaluate if composite anomalies significantly differ from zero. This step includes:
* **Effective Sample Size (ESS):** An inherent correction that penalizes the nominal sample size proportionally to the lag-1 autocorrelation coefficient to account for serial persistence.
* **Bonferroni Correction:** A conservative multi-variable adjustment to control for Type-I error inflation across multiple regimes and regions.

### D. Joint Probability Density (KDE) Analysis
Run `joint_probability_and_bootstrap.py` alongside the derived `regime_results.npz` file. This script mathematically constructs the joint probability distributions of the standardized anomalies and derives the 95% confidence intervals for the compound deficit probabilities through statistical bootstrapping.
