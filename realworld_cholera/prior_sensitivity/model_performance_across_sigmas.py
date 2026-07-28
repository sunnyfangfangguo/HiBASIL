# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 17:24:35 2026

@author: sunny
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sigmas = [5, 10, 25, 50, 100, 250]
model_types = ['exponential', 'gaussian', 'power_law']


"summarize model selection"
sigam_ls = []
models = []
rank_loos = []
elpd_loos = []
p_loos = []
se_d_loos = []
aics = []
bics = []
delta_elpds = [] # differ to ELPD of the rank 0 model

for sigma in sigmas:
    modelperfomance_file = f"./sigma{sigma}/output/models_performances.csv"
    
    performance_df = pd.read_csv(modelperfomance_file, index_col=0)
    
    for model_type in model_types:
        rank_loo = performance_df.at['rank_loo', model_type]
        elpd_loo = performance_df.at['elpd_loo', model_type]
        p_loo = performance_df.at['p_loo', model_type]
        se_d_loo = performance_df.at['se_d_loo', model_type]
        
        rank0_model = performance_df.columns[performance_df.loc['rank_loo'] == '0'].tolist()[0]
        rank0_elpd = performance_df.at['elpd_loo', rank0_model]
        delta_elpd = float(elpd_loo) - float(rank0_elpd)
        
        sigam_ls.append(sigma)
        models.append(model_type)
        rank_loos.append(rank_loo)
        elpd_loos.append(elpd_loo)
        p_loos.append(p_loo)
        se_d_loos.append(se_d_loo)
        delta_elpds.append(delta_elpd)
        
model_select_df = pd.DataFrame({'sigma':sigam_ls, 'Model':models, 'rank_loo':rank_loos,
                                'elpd_loo':elpd_loos, 'p_loo':p_loos, 'se_d_loo':se_d_loos,
                                'delta_elpd':delta_elpds}) 

# delta ELPD below the threshold of significance (< 2* SE), effectively treating models as predictive equivalents 
model_select_df['Predictive Equivalent'] = (abs(model_select_df['delta_elpd']) <= 2 * model_select_df['se_d_loo'].astype(float))

model_select_df.to_csv("model_selection_summary_sigmas.csv")
    

"calculate models localization accuracy and discrimination"
def calc_discrimination_ratio(pumps_df, fx1_est, fy1_est, fx1_true, fy1_true):
    " Discrimination Ratio = Distance to Nearest Alternative / Distance to True Source"
    true_source = 'Broad St Pump'
    
    # Calculate distance to true source
    dist_to_true = np.sqrt((fx1_est - fx1_true)**2 + 
                           (fy1_est - fy1_true)**2)
    
    # Calculate distances to all alternative sources
    distances_to_alternatives = {}
    for idx, row in pumps_df.iterrows():
        pump_name = row['name']
        px = row['COORD_X']
        py = row['COORD_Y']
        if pump_name != true_source:  # Skip the true source
            dist = np.sqrt((fx1_est - px)**2 + 
                          (fy1_est - py)**2)
            distances_to_alternatives[pump_name] = dist
    
    # Find nearest alternative
    nearest_alt = min(distances_to_alternatives, 
                     key=distances_to_alternatives.get)
    dist_to_nearest_alt = distances_to_alternatives[nearest_alt]
    
    # Calculate discrimination ratio
    ratio = dist_to_nearest_alt / dist_to_true if dist_to_true > 0 else np.inf
    
    return dist_to_true, dist_to_nearest_alt, nearest_alt, ratio



pumps_df = pd.read_csv("pumps.csv")
Broad_St_pump_location = [529396.5394, 181025.063]
fx1_true = Broad_St_pump_location[0]
fy1_true = Broad_St_pump_location[1]

sigam_ls = []
models = []
fx1_trues = []
fy1_trues = []
fx1_ests = []
fy1_ests = []

distance_to_trues = []
distance_to_nearest_alts = []
nearest_alternatives = []
discrimination_ratios = []

for sigma in sigmas:
    for model_type in model_types:
        tracesummary_file = f"./sigma{sigma}/output/{model_type}_trace_summary.csv"    
        tracesummary_df = pd.read_csv(tracesummary_file, index_col=0)
        fx1_est = tracesummary_df.at['fx1', 'mean']
        fy1_est = tracesummary_df.at['fy1', 'mean']
        
        "calculate discrimination"
        distance_to_true, distance_to_nearest_alt, nearest_alternative, discrimination_ratio = calc_discrimination_ratio(pumps_df, fx1_est, fy1_est, fx1_true, fy1_true) 

        "store values"
        sigam_ls.append(sigma)
        models.append(model_type)
        fx1_trues.append(fx1_true)
        fy1_trues.append(fy1_true)
        fx1_ests.append(fx1_est)
        fy1_ests.append(fy1_est)
        
        distance_to_trues.append(distance_to_true)
        distance_to_nearest_alts.append(distance_to_nearest_alt)
        nearest_alternatives.append(nearest_alternative)
        discrimination_ratios.append(discrimination_ratio)
        
        
params_est_df = pd.DataFrame({'sigma':sigam_ls, 'Model':models, 'fx1_true':fx1_trues,
                                'fy1_true':fy1_trues, 'fx1_est':fx1_ests, 
                                'fy1_est':fy1_ests, 'localization_error': distance_to_trues,
                                'distance_to_nearest_alternative':distance_to_nearest_alts,
                                'nearest_alternative':nearest_alternatives,
                                'discrimination_ratio':discrimination_ratios
                                })         
        
        
params_est_df.to_csv("parameter_estimation_localization_errors_sigmas.csv") 



"plot sigmas with localization errors"
# Professional seaborn settings
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)

# Colorblind-safe colors
colors = {
    'powerlaw': '#0072B2',    # Blue
    'plateau': '#009E73'      # Green
}

# Data (Power Law only)
sigma = params_est_df[params_est_df['Model']=='power_law']['sigma']
error = params_est_df[params_est_df['Model']=='power_law']['localization_error']

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plateau highlighting (σ≥50m)
ax.axhspan(31, 33.5, xmin=0.38, xmax=1.0, 
           alpha=0.12, color=colors['plateau'], 
           label='Plateau region (σ≥50m)', zorder=1)

# Main sensitivity curve
ax.plot(sigma, error, 'o-', linewidth=3.5, markersize=13,
        color=colors['powerlaw'], markerfacecolor=colors['powerlaw'],
        markeredgewidth=2.5, markeredgecolor='white', zorder=3)

# Plateau indicator line
ax.axvline(x=50, color='gray', linestyle=':', 
           linewidth=2, alpha=0.5, zorder=2)

# Annotation for plateau
ax.text(52, 35.5, 'Plateau begins', fontsize=12, 
        color=colors['plateau'], style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor=colors['plateau'], alpha=0.9, linewidth=1.5))

# Key point annotations (minimal)
ax.annotate('7.5m', xy=(5, 7.510), xytext=(5, 4),
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.annotate('32.7m', xy=(250, 32.739), xytext=(250, 36),
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Labels
ax.set_xlabel('Prior Standard Deviation σ (m)', 
              fontsize=15, fontweight='bold')
ax.set_ylabel('Localization Error (m)', 
              fontsize=15, fontweight='bold')

# Axis limits
ax.set_xlim(0, 265)
ax.set_ylim(0, 40)

# Grid
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)

# Legend
ax.legend(loc='lower right', fontsize=15, framealpha=0.98, 
          edgecolor='darkgray', fancybox=True)

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout
plt.tight_layout()

# Save
plt.savefig('prior_sensitivity_powerlaw.png', dpi=600, bbox_inches='tight', facecolor='white')





    
    
