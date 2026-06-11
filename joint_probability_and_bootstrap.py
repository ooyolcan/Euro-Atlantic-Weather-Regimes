"""
SUPPLEMENTARY MATERIAL:
Bivariate Joint Probability Density and Bootstrapping Pipeline

This script generates 2D Kernel Density Estimation (KDE) plots for any two 
environmental variables and computes the 'Compound Deficit' probability 
(the lower-left quadrant where both variables are below their climatological zero).
It also includes a rigorous 1000-iteration bootstrapping procedure to derive 
the 95% Confidence Intervals for these joint probabilities.

INSTRUCTIONS FOR USERS:
- Replace 'variable_1_name' and 'variable_2_name' with your actual target datasets.
- Define your regional bounding box in the configuration section.
- Ensure you have a cluster/regime labels array matching the time dimension of your data.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. USER CONFIGURATION BLOCK
# ==============================================================================
# Define paths to your specific datasets
NETCDF_DATA_PATH = "path/to/your_daily_data.nc"
REGIME_LABELS_PATH = "path/to/your_cluster_labels.npz"

# Define the two variables for joint probability analysis (e.g., wind and solar)
VAR_X = "variable_1_name"  
VAR_Y = "variable_2_name"

# Define target regional boundaries (Latitude/Longitude)
LAT_MIN, LAT_MAX = 00.0, 00.0
LON_MIN, LON_MAX = 00.0, 00.0

N_BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 95
# ==============================================================================

# ==============================================================================
# 2. DATA LOADING & REGIONAL SLICING
# ==============================================================================
print("Loading datasets and extracting target region...")

# Example data loading (Users should adapt this to their specific data structures)
try:
    ds = xr.open_dataset(NETCDF_DATA_PATH)
    labels_data = np.load(REGIME_LABELS_PATH, allow_pickle=True)
    regime_labels = labels_data['labels']
except Exception as e:
    print("Please ensure you have linked the correct data paths in the configuration block.")
    print(f"Error detail: {e}")
    # A dummy dataframe is created below just to demonstrate the logic if files are missing
    print("Generating dummy data for methodological demonstration...")
    np.random.seed(42)
    n_samples = 5000
    regime_labels = np.random.choice([1, 2, 3, 4], size=n_samples)
    dummy_x = np.random.normal(0, 1, n_samples)
    dummy_y = np.random.normal(0, 1, n_samples)
    df = pd.DataFrame({'Regime': regime_labels, 'Var_X_Anom': dummy_x, 'Var_Y_Anom': dummy_y})

# NOTE: If using real NetCDF data, users should extract their regional anomalies here
# using a cyclic 5-day smoothing methodology similar to the main clustering pipeline.
# For demonstration purposes, we assume 'df' contains the calculated anomalies.

# ==============================================================================
# 3. KDE PLOTTING & BOOTSTRAPPED CONFIDENCE INTERVALS
# ==============================================================================
print("\nInitiating KDE plotting and Bootstrapping Procedure...\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = ['Blues', 'Purples', 'Oranges', 'Greens']

unique_regimes = sorted(list(set(regime_labels)))

for i, r in enumerate(unique_regimes):
    subset = df[df['Regime'] == r]
    n_subset = len(subset)
    
    x_anom = subset['Var_X_Anom'].values
    y_anom = subset['Var_Y_Anom'].values
    
    # --- KDE Plot Generation ---
    sns.kdeplot(x=x_anom, y=y_anom, ax=axes[i], fill=True, cmap=colors[i], thresh=0.05, alpha=0.8)
    
    # Zero-anomaly thresholds
    axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[i].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    axes[i].set_title(f'Cluster/Regime {r}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel(f'{VAR_X} Anomaly', fontsize=11)
    axes[i].set_ylabel(f'{VAR_Y} Anomaly', fontsize=11)
    
    # --- Original Compound Deficit Calculation ---
    # Counting days where BOTH anomalies are strictly negative
    deficit_mask = (x_anom < 0) & (y_anom < 0)
    original_prob = (np.sum(deficit_mask) / n_subset) * 100
    
    # --- Bootstrapping for Confidence Intervals ---
    bootstrapped_probs = []
    
    # Set a local seed for reproducibility within the bootstrap loop
    rng = np.random.default_rng(42)
    
    for _ in range(N_BOOTSTRAP_ITERATIONS):
        # Resampling with replacement
        boot_indices = rng.choice(n_subset, size=n_subset, replace=True)
        sample_x = x_anom[boot_indices]
        sample_y = y_anom[boot_indices]
        
        prob = np.sum((sample_x < 0) & (sample_y < 0)) / n_subset
        bootstrapped_probs.append(prob)
        
    lower_percentile = (100 - CONFIDENCE_LEVEL) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_bound = np.percentile(bootstrapped_probs, lower_percentile) * 100
    upper_bound = np.percentile(bootstrapped_probs, upper_percentile) * 100
    
    # Displaying results
    print(f"--- Regime {r} ---")
    print(f"Analyzed Days : {n_subset}")
    print(f"Original Prob : {original_prob:.1f}%")
    print(f"95% CI Range  : [{lower_bound:.1f}% - {upper_bound:.1f}%]\n")

    # Annotate plot
    axes[i].text(0.05, 0.05, f'Deficit: {original_prob:.1f}%\nCI: [{lower_bound:.1f}%-{upper_bound:.1f}%]', 
                 transform=axes[i].transAxes, fontsize=10, color='darkred', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))

plt.suptitle('Joint Probability Density of Anomalies with Confidence Intervals', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save output
output_filename = "Anonymous_KDE_Plot.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully as '{output_filename}'.")
plt.show()
