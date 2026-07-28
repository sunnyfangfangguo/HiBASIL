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


output_dir = './'
base_csv_path = output_dir + 'model_selection_summary_base.csv'
narrowsteep_csv_path = output_dir + 'model_selection_summary_narrowsteep.csv'
wideshallow_csv_path = output_dir + 'model_selection_summary_wideshallow.csv'
widesteep_csv_path = output_dir + 'model_selection_summary_widesteep.csv'


base_df = pd.read_csv(base_csv_path)
narrowsteep_df = pd.read_csv(narrowsteep_csv_path)
wideshallow_df = pd.read_csv(wideshallow_csv_path)
widesteep_df = pd.read_csv(widesteep_csv_path)

# Define the order of kernels on the X-axis
model_order = ['power_law', 'exponential', 'gaussian']

# 3. Calculate the average SE for the "Equivalence Zone"
# This defines the gray shaded area where models are 'tied'
base_avg_se = base_df[base_df['Model'] != 'power_law']['se_d_loo'].mean()
base_equiv_threshold = 2 * base_avg_se

narrowsteep_avg_se = narrowsteep_df[narrowsteep_df['Model'] != 'power_law']['se_d_loo'].mean()
narrowsteep_equiv_threshold = 2 * narrowsteep_avg_se

wideshallow_avg_se = wideshallow_df[wideshallow_df['Model'] != 'power_law']['se_d_loo'].mean()
wideshallow_equiv_threshold = 2 * wideshallow_avg_se

widesteep_avg_se = widesteep_df[widesteep_df['Model'] != 'power_law']['se_d_loo'].mean()
widesteep_equiv_threshold = 2 * widesteep_avg_se


"Plot"
fig, axs = plt.subplots(2,2,figsize=(10,10))

"Base result"
# Draw the boxplot (the main distribution)
sns.boxplot(x='Model', y='delta_elpd', data=base_df, order=model_order, 
    palette='viridis', width=0.5, hue='Model',
    fliersize=0,  # Hide outliers here to avoid double-plotting with stripplot
    ax=axs[0][0])

# Overlay individual simulation points (stripplot) for transparency
sns.stripplot(x='Model', y='delta_elpd', data=base_df, order=model_order, 
    color='black', alpha=0.3, jitter=True, size=4, ax=axs[0][0])

# 5. Add Reference Lines
# Red dashed line for the winner (at 0)
axs[0][0].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Best Model (Power Law)')

# Shaded gray zone for Predictive Equivalence (2 * SE)
axs[0][0].axhspan(-base_equiv_threshold, 0, color='gray', alpha=0.2, label='Equivalence Zone (2×SE)')

# 6. Formatting and Labels
axs[0][0].text(-0.15, 1.05, 'A', transform=axs[0][0].transAxes,
            fontsize=16, fontweight='bold', va='top')

axs[0][0].set_ylabel(r'$\Delta ELPD_{loo}$ (relative to winner)', fontsize=12)
axs[0][0].set_xlabel('Dispersal Kernel', fontsize=12)

# Adjust limits to ensure the gap is the focus
axs[0][0].set_ylim(base_df['delta_elpd'].min() - 50, 50)
axs[0][0].legend(loc='lower left', frameon=True)

current_ticks = axs[0][0].get_xticks()
axs[0][0].set_xticks(current_ticks, labels=['Power Law', 'Exponential', 'Gaussian'])

"narrowsteep result"
# Draw the boxplot (the main distribution)
sns.boxplot(x='Model', y='delta_elpd', data=narrowsteep_df, order=model_order, 
    palette='viridis', width=0.5, hue='Model',
    fliersize=0,  # Hide outliers here to avoid double-plotting with stripplot
    ax=axs[0][1])

# Overlay individual simulation points (stripplot) for transparency
sns.stripplot(x='Model', y='delta_elpd', data=narrowsteep_df, order=model_order, 
    color='black', alpha=0.3, jitter=True, size=4, ax=axs[0][1])

# 5. Add Reference Lines
# Red dashed line for the winner (at 0)
axs[0][1].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Best Model (Power Law)')

# Shaded gray zone for Predictive Equivalence (2 * SE)
axs[0][1].axhspan(-narrowsteep_equiv_threshold, 0, color='gray', alpha=0.2, label='Equivalence Zone (2×SE)')

# 6. Formatting and Labels
axs[0][1].text(-0.15, 1.05, 'B', transform=axs[0][1].transAxes,
            fontsize=16, fontweight='bold', va='top')

axs[0][1].set_ylabel(r'$\Delta ELPD_{loo}$ (relative to winner)', fontsize=12)
axs[0][1].set_xlabel('Dispersal Kernel', fontsize=12)

# Adjust limits to ensure the gap is the focus
axs[0][1].set_ylim(narrowsteep_df['delta_elpd'].min() - 50, 50)
axs[0][1].legend(loc='lower left', frameon=True)

current_ticks = axs[0][1].get_xticks()
axs[0][1].set_xticks(current_ticks, labels=['Power Law', 'Exponential', 'Gaussian'])



"widesteep result"
# Draw the boxplot (the main distribution)
sns.boxplot(x='Model', y='delta_elpd', data=widesteep_df, order=model_order, 
    palette='viridis', width=0.5, hue='Model',
    fliersize=0,  # Hide outliers here to avoid double-plotting with stripplot
    ax=axs[1][0])

# Overlay individual simulation points (stripplot) for transparency
sns.stripplot(x='Model', y='delta_elpd', data=widesteep_df, order=model_order, 
    color='black', alpha=0.3, jitter=True, size=4, ax=axs[1][0])

# 5. Add Reference Lines
# Red dashed line for the winner (at 0)
axs[1][0].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Best Model (Power Law)')

# Shaded gray zone for Predictive Equivalence (2 * SE)
axs[1][0].axhspan(-widesteep_equiv_threshold, 0, color='gray', alpha=0.2, label='Equivalence Zone (2×SE)')

# 6. Formatting and Labels
axs[1][0].text(-0.15, 1.05, 'C', transform=axs[1][0].transAxes,
            fontsize=16, fontweight='bold', va='top')

axs[1][0].set_ylabel(r'$\Delta ELPD_{loo}$ (relative to winner)', fontsize=12)
axs[1][0].set_xlabel('Dispersal Kernel', fontsize=12)

# Adjust limits to ensure the gap is the focus
axs[1][0].set_ylim(widesteep_df['delta_elpd'].min() - 50, 50)
axs[1][0].legend(loc='lower left', frameon=True)

current_ticks = axs[1][0].get_xticks()
axs[1][0].set_xticks(current_ticks, labels=['Power Law', 'Exponential', 'Gaussian'])



"wideshallow result"
# Draw the boxplot (the main distribution)
sns.boxplot(x='Model', y='delta_elpd', data=wideshallow_df, order=model_order, 
    palette='viridis', width=0.5, hue='Model',
    fliersize=0,  # Hide outliers here to avoid double-plotting with stripplot
    ax=axs[1][1])

# Overlay individual simulation points (stripplot) for transparency
sns.stripplot(x='Model', y='delta_elpd', data=wideshallow_df, order=model_order, 
    color='black', alpha=0.3, jitter=True, size=4, ax=axs[1][1])

# 5. Add Reference Lines
# Red dashed line for the winner (at 0)
axs[1][1].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Best Model (Power Law)')

# Shaded gray zone for Predictive Equivalence (2 * SE)
axs[1][1].axhspan(-wideshallow_equiv_threshold, 0, color='gray', alpha=0.2, label='Equivalence Zone (2×SE)')

# 6. Formatting and Labels
axs[1][1].text(-0.15, 1.05, 'D', transform=axs[1][1].transAxes,
            fontsize=16, fontweight='bold', va='top')

axs[1][1].set_ylabel(r'$\Delta ELPD_{loo}$ (relative to winner)', fontsize=12)
axs[1][1].set_xlabel('Dispersal Kernel', fontsize=12)

# Adjust limits to ensure the gap is the focus
axs[1][1].set_ylim(wideshallow_df['delta_elpd'].min() - 50, 50)
axs[1][1].legend(loc='lower left', frameon=True)

current_ticks = axs[1][1].get_xticks()
axs[1][1].set_xticks(current_ticks, labels=['Power Law', 'Exponential', 'Gaussian'])



# 7. Save and Show
plt.tight_layout()
plt.savefig(output_dir + 'dispersal_scenarios_delta_elpd_plot.png', dpi=300)



