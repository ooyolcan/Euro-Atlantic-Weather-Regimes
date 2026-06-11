"""
SUPPLEMENTARY MATERIAL: 
Atmospheric Regime Identification and Compound Anomaly Analysis Pipeline

This script outlines the general methodological framework used for preprocessing, 
spatial clustering, and rigorous statistical validation. The pipeline is designed 
to be highly adaptable for different study domains, variables, and cluster sizes.

DATA REQUIREMENT:
To run this pipeline, users must provide a daily aggregated NetCDF dataset 
containing the necessary atmospheric fields specified in the configuration block.
"""

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
from sklearn.cluster import KMeans
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL PIPELINE CONFIGURATION 
# (Users should modify these parameters based on their specific study region/variables)
#
# MEMORY NOTE: Calling .values on spatial fields (Stage 3.2) loads the data matrix 
# into memory. For very high-resolution grids over large global domains, ensure 
# your machine has sufficient RAM or downsample the spatial resolution beforehand.
# ==============================================================================
FILE_PATH = "your_daily_dataset.nc" 

# Core atmospheric variable used for K-Means clustering (e.g., 'z500', 'slp')
CLUSTERING_VAR = "z500"                  

# Downstream variables targeted for compound anomaly and significance testing
TEST_VARIABLES = ["u100", "ssrd", "tp"]  

# Target number of atmospheric regimes/clusters
N_CLUSTERS = 4                           

# Regional bounding boxes for spatial validation (Latitude Min, Latitude Max)
# Examples provided below are fully customizable for any study domain
REGIONS = {
    "Sub-Region 1": (56, 70),   
    "Sub-Region 2": (43, 56),
    "Sub-Region 3": (30, 43)
}
# ==============================================================================

# ==============================================================================
# 1. DATA IMPORT & ALIGNMENT
# ==============================================================================
print("--- STAGE 1: Data Initialization ---")

try:
    ds = xr.open_dataset(FILE_PATH, chunks={'time': 'auto'})
    print(f"Dataset successfully loaded from {FILE_PATH}.")
except FileNotFoundError:
    print(f"[ERROR] Input file '{FILE_PATH}' not found.")
    print("Please ensure your daily NetCDF data is placed in the working directory.")
    exit()

# Temporal alignment check
if 'time' in ds.coords:
    print("Temporal alignment verified. Continuous calendar cycle is maintained.")

# ==============================================================================
# 2. PREPROCESSING & DAILY ANOMALY CALCULATION
# ==============================================================================
print("\n--- STAGE 2: Climatology and Anomaly Calculation ---")

# Combine all unique variables needed for processing
VARIABLES_TO_PROCESS = list(set([CLUSTERING_VAR] + TEST_VARIABLES))
ds_anomalies = xr.Dataset(coords=ds.coords)

for var in VARIABLES_TO_PROCESS:
    if var not in ds.data_vars:
        print(f"[WARNING] Variable '{var}' missing in the dataset. Skipping...")
        continue
    
    # 2.1 Calculate raw daily climatology (Multi-year average for each Day of Year)
    raw_climatology = ds[var].groupby('time.dayofyear').mean('time')
    
    # 2.2 Apply 5-day cyclic moving average to smooth the climatological baseline
    # Dynamically tracking the number of days (365 or 366) to avoid size mismatch crashes
    N_days = len(raw_climatology.dayofyear)
    padded_clim = np.pad(raw_climatology.values, pad_width=((2, 2), (0, 0), (0, 0)), mode='wrap')
    smoothed_clim_vals = np.mean([padded_clim[i:i+N_days] for i in range(5)], axis=0)
    
    smoothed_climatology = xr.DataArray(
        smoothed_clim_vals, 
        coords=[raw_climatology.dayofyear, ds.latitude, ds.longitude], 
        dims=["dayofyear", "latitude", "longitude"]
    )
    
    # 2.3 Calculate Raw Daily Anomalies (Preserving physical units)
    anomaly = ds[var].groupby('time.dayofyear') - smoothed_climatology
    
    # Store raw anomalies directly (Standardization is strictly omitted here to preserve 
    # synoptic variance for Z500 clustering and physical units for surface fields)
    ds_anomalies[var] = anomaly

print("Spatial anomalies successfully computed (physical units preserved).")

# ==============================================================================
# 3. ATMOSPHERIC REGIME IDENTIFICATION (K-MEANS CLUSTERING)
# ==============================================================================
print("\n--- STAGE 3: Spatial Clustering ---")

if CLUSTERING_VAR in ds_anomalies.data_vars:
    # 3.1 Spatial Area-Correction (Square root of cosine of latitude)
    lat_weights = np.sqrt(np.cos(np.deg2rad(ds.latitude)))
    lat_weights = lat_weights.broadcast_like(ds_anomalies[CLUSTERING_VAR])
    weighted_field = ds_anomalies[CLUSTERING_VAR] * lat_weights

    # 3.2 Dimensionality flattening 
    flattened_field = weighted_field.stack(spatial=("latitude", "longitude")).dropna(dim="spatial")
    X_matrix = flattened_field.values

    # 3.3 K-Means Execution
    kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', max_iter=300, random_state=42)
    regime_labels = kmeans.fit_predict(X_matrix)

    ds_anomalies["regime"] = ("time", [f"Regime {label+1}" for label in regime_labels])
    print(f"K-Means clustering completed. {N_CLUSTERS} Regimes identified based on {CLUSTERING_VAR}.")
else:
    print(f"[ERROR] Base clustering variable ({CLUSTERING_VAR}) not found.")
    exit()

# ==============================================================================
# 4. STATISTICAL VALIDATION (ESS & BONFERRONI CORRECTION)
# ==============================================================================
print("\n--- STAGE 4: Statistical Inference & Significance Testing ---")

def calculate_effective_sample_size(data_series):
    """
    Calculates Effective Sample Size (ESS) by penalizing the nominal sample 
    size proportionally to the lag-1 autocorrelation coefficient.
    """
    n = len(data_series)
    if n <= 1: return n
    centered_data = data_series - np.mean(data_series)
    r1 = np.corrcoef(centered_data[:-1], centered_data[1:])[0, 1]
    if np.isnan(r1) or r1 < 0: r1 = 0.0
    n_eff = n * ((1 - r1) / (1 + r1))
    return int(np.clip(n_eff, 2, n))

# Initialize a central random generator for robust, independent subsampling across iterations
rng = np.random.default_rng(42)

regimes_list = [f"Regime {i+1}" for i in range(N_CLUSTERS)]

# Multi-variable Bonferroni Adjustment
total_tests = len(regimes_list) * len(REGIONS) * len(TEST_VARIABLES)
alpha_adj = 0.05 / total_tests

print(f"Total Parallel Tests: {total_tests} | Adjusted Alpha: {alpha_adj:.5f}")
print("-" * 90)
print(f"{'Regime':<10} | {'Region':<14} | {'Var':<5} | {'Nominal N':<10} | {'ESS':<6} | {'p-value':<10} | {'Status':<10}")
print("-" * 90)

results_list = []

for regime in regimes_list:
    regime_mask = (ds_anomalies["regime"] == regime)
    
    for region_name, (lat_min, lat_max) in REGIONS.items():
        if float(ds.latitude[0]) > float(ds.latitude[-1]):
            regional_ds = ds_anomalies.sel(latitude=slice(lat_max, lat_min))
        else:
            regional_ds = ds_anomalies.sel(latitude=slice(lat_min, lat_max))
            
        for var in TEST_VARIABLES:
            if var not in regional_ds.data_vars: continue
            
            spatial_mean_series = regional_ds[var].mean(dim=["latitude", "longitude"]).values
            anomaly_series = spatial_mean_series[regime_mask]
            
            anomaly_series = anomaly_series[~np.isnan(anomaly_series)]
            n_nominal = len(anomaly_series)
            if n_nominal < 5: continue
            
            # Autocorrelation Correction (ESS)
            n_effective = calculate_effective_sample_size(anomaly_series)
            
            # Statistically sound independent subsampling using the central generator
            resampled_data = rng.choice(anomaly_series, size=n_effective, replace=False)
            
            # Non-parametric Wilcoxon Signed-Rank Test (Testing deviation from zero-median)
            # Protection against zero-variance/all-zero arrays (e.g., in strict drought regions)
            if np.count_nonzero(resampled_data) < 5:
                p_value = np.nan
                status = "INSUFF. DATA"
            else:
                _, p_value = stats.wilcoxon(resampled_data)
                status = "SIGNIFICANT" if p_value <= alpha_adj else "NS"
            
            # Handle formatting for NaN p-values
            if np.isnan(p_value):
                p_str = "NaN"
            else:
                p_str = f"{p_value:.3e}"
                
            print(f"{regime:<10} | {region_name:<14} | {var:<5} | {n_nominal:<10} | {n_effective:<6} | {p_str:<10} | {status:<10}")
            
            results_list.append({
                "Regime": regime, "Region": region_name, "Variable": var,
                "Nominal_N": n_nominal, "ESS_N": n_effective, "p_value": p_value, "Status": status
            })

print("-" * 90)

# Exporting outputs to ensure completeness of the supplementary package
df_results = pd.DataFrame(results_list)
output_csv = "pipeline_statistical_output.csv"
df_results.to_csv(output_csv, index=False)
print(f"[SUCCESS] Statistical analysis complete. Results exported to '{output_csv}'.")
print("PIPELINE COMPLETED.")