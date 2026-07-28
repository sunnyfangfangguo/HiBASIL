# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:19:17 2025

@author: sunny
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

"posterior predictive performace metrics"



sns.set_theme(style="darkgrid")
output_dir = './'
models_to_test = ['exponential', 'gaussian', 'power_law']

sample_sizes = ['known_onefocus', 'known_twofoci', 
                'unknown_onefocus', 'unknown_twofoci']

model_order = ['power_law', 'exponential', 'gaussian']


for size in sample_sizes:
    metrics_df = pd.read_csv(f'{output_dir}ppc_metrics_all_sims_{size}.csv')
    grouped_df = metrics_df.groupby('model_type')
    grouped_df = grouped_df[['overall_r2','overall_rmse','overall_mae','obs_zeros',
                       'pred_zeros', 'obs_ones', 'pred_ones', 'continuous_r2',
                       'continuous_rmse']].agg(['mean', 'std'])
    
    grouped_df.to_csv(f'{output_dir}posterior_predictive_performace_metrics_summary_{size}.csv')

for model_type in models_to_test:
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    labels = ['A', 'B', 'C', 'D',
              'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L',
              'M', 'N', 'O', 'P'
              ]
    
    for idx, size in enumerate(sample_sizes):
        metrics_df = pd.read_csv(f'{output_dir}ppc_metrics_all_sims_{size}.csv')

        model_metrics_df = metrics_df[metrics_df['model_type']==model_type]
                
        "overall metrics"
        sub1_df = model_metrics_df[['model_type', 'overall_r2', 'overall_rmse', 'overall_mae']].set_index('model_type')
        sub1_df_stacked = sub1_df.stack().to_frame().reset_index()
        sub1_df_stacked.columns = ['model_type', 'metric', 'value']
        sns.boxplot(data=sub1_df_stacked, y='value', hue='metric', ax=axes[idx][0])
        
        label_seq = idx * 4 + 0
        label = labels[label_seq]
        axes[idx][0].text(-0.15, 1.05, label, transform=axes[idx][0].transAxes, fontsize=14, fontweight='bold')
        
        "continuous component metric"
        sub2_df = model_metrics_df[['model_type', 'continuous_r2', 'continuous_rmse']].set_index('model_type')
        sub2_df_stacked = sub2_df.stack().to_frame().reset_index()
        sub2_df_stacked.columns = ['model_type', 'metric', 'value']
        sns.boxplot(data=sub2_df_stacked, y='value', hue='metric', ax=axes[idx][1])
        
        label_seq = idx * 4 + 1
        label = labels[label_seq]
        axes[idx][1].text(-0.15, 1.05, label, transform=axes[idx][1].transAxes, fontsize=14, fontweight='bold')
        
        "zero-inflation component"
        sns.regplot(data=model_metrics_df, x='obs_zeros', y='pred_zeros', ax=axes[idx][2])
        zero_pearson_r, zero_p_value = stats.pearsonr(model_metrics_df['obs_zeros'], model_metrics_df['pred_zeros'])
        zero_pearson_r2 = zero_pearson_r ** 2
        zero_stats_text = f'$R^2$ = {zero_pearson_r2:.2f}\nP-val = {zero_p_value:.2f}'
        axes[idx][2].text(0.10, 0.95, zero_stats_text, transform=axes[idx][2].transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        label_seq = idx * 4 + 2
        label = labels[label_seq]
        axes[idx][2].text(-0.15, 1.05, label, transform=axes[idx][2].transAxes, fontsize=14, fontweight='bold')
        axes[idx][2].set_xlabel("Observed zero proportion")
        axes[idx][2].set_ylabel("Predicted zero proportion")
        
        "one-inflation component"
        sns.regplot(data=model_metrics_df, x='obs_ones', y='pred_ones', ax=axes[idx][3])
        one_pearson_r, one_p_value = stats.pearsonr(model_metrics_df['obs_ones'], model_metrics_df['pred_ones'])
        one_pearson_r2 = one_pearson_r ** 2
        one_stats_text = f'$R^2$ = {one_pearson_r2:.2f}\nP-val = {one_p_value:.2f}'
        axes[idx][3].text(0.10, 0.95, one_stats_text, transform=axes[idx][3].transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        label_seq = idx * 4 + 3
        label = labels[label_seq]
        axes[idx][3].text(-0.15, 1.05, label, transform=axes[idx][3].transAxes, fontsize=14, fontweight='bold')
        axes[idx][3].set_xlabel("Observed one proportion")
        axes[idx][3].set_ylabel("Predicted one proportion")
            
            
    plt.tight_layout()
    plt.savefig(f'{output_dir}{model_type}_known_unknown_posterior_predictive_check.png',
                dpi=600, bbox_inches='tight')






