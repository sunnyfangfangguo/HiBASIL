# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 17:23:10 2025

@author: sunny
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_theme(style="whitegrid")


scenarios = ['narrowsteep8', 'wideshallow75']
model_order = ['power_law', 'exponential', 'gaussian']
labels = ['A', 'B']

output_dir = './'# Define the order of kernels on the X-axis

fig, axs = plt.subplots(1,2,figsize=(10,5))
for idx, scenario in enumerate(scenarios):
    csv_path = f"{output_dir}model_selection_summary_{scenario}.csv"
    df = pd.read_csv(csv_path)

    # 3. Calculate the average SE for the "Equivalence Zone"
    # This defines the gray shaded area where models are 'tied'
    avg_se = df[df['Model'] != 'power_law']['se_d_loo'].mean()
    equiv_threshold = 2 * avg_se

    col = idx % 2
    
    # Draw the boxplot (the main distribution)
    sns.boxplot(x='Model', y='delta_elpd', data=df, order=model_order, 
        palette='viridis', width=0.5, hue='Model',
        fliersize=0,  # Hide outliers here to avoid double-plotting with stripplot
        ax=axs[col])
    
    # Overlay individual simulation points (stripplot) for transparency
    sns.stripplot(x='Model', y='delta_elpd', data=df, order=model_order, 
        color='black', alpha=0.3, jitter=True, size=4, ax=axs[col])
    
    # 5. Add Reference Lines
    # Red dashed line for the winner (at 0)
    axs[col].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Best Model (Power Law)')
    
    # Shaded gray zone for Predictive Equivalence (2 * SE)
    axs[col].axhspan(-equiv_threshold, 0, color='gray', alpha=0.2, label='Equivalence Zone (2×SE)')
    
    # 6. Formatting and Labels
    label = labels[idx]
    axs[col].text(-0.15, 1.05, label, transform=axs[col].transAxes,
                fontsize=16, fontweight='bold', va='top')
    
    axs[col].set_ylabel(r'$\Delta ELPD_{loo}$ (relative to winner)', fontsize=12)
    axs[col].set_xlabel('Dispersal Kernel', fontsize=12)
    
    # Adjust limits to ensure the gap is the focus
    axs[col].set_ylim(df['delta_elpd'].min() - 50, 50)
    axs[col].legend(loc='lower left', frameon=True)
    
    current_ticks = axs[col].get_xticks()
    axs[col].set_xticks(current_ticks, labels=['Power Law', 'Exponential', 'Gaussian'])
    
    
# 7. Save and Show
plt.tight_layout()
plt.savefig(output_dir + 'supplementary_dispersal_scenarios_delta_elpd_plot.png', dpi=600)

    
    
