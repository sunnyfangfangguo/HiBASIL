# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:17:19 2025

@author: sunny
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_posterior_means_distribution(metrics_df, models_to_test, n_sims):
    """
    Plot distribution of posterior means across simulations.
    Shows if posteriors consistently stay near prior.
    """
    
    for model_type in models_to_test:
        # Load metrics
        model_metrics_df = metrics_df[metrics_df['model_type']==model_type]
    
        param_names = ['scale1']
        param_labels = ['Scale']
        if model_type == 'power_law':
            param_names.append('exponent1')
            param_labels.append('Exponent')
        
        label_list = ['A', 'B']
        n_params = len(param_names)
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
        if n_params == 1:
            axes = [axes]
        
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            
            # Get posterior means from all simulations
            post_means = model_metrics_df[f'{param}_post_mean'].values
            prior_mean = model_metrics_df[f'{param}_prior_mean'].values[0]
            prior_std = model_metrics_df[f'{param}_prior_std'].values[0]
            
            # Histogram of posterior means
            ax.hist(post_means, bins=30, alpha=0.6, color='blue', 
                   density=True, label=f'Posterior means (n={n_sims})')
            
            # Prior mean reference
            ax.axvline(prior_mean, color='black', linewidth=3, linestyle='--',
                      label=f'Prior mean = {prior_mean:.2f}')
            
            # Prior ± 1 SD
            ax.axvspan(prior_mean - prior_std, prior_mean + prior_std,
                      alpha=0.4, color='lightgray', label='Prior ± 1 SD')
            
            # Statistics
            mean_of_means = post_means.mean()
            std_of_means = post_means.std()
            
            ax.axvline(mean_of_means, color='blue', linewidth=2,
                      label=f'Mean of posteriors = {mean_of_means:.2f}')
            
            # Calculate % within prior ±1 SD
            within_prior = np.mean((post_means >= prior_mean - prior_std) & 
                                  (post_means <= prior_mean + prior_std))
            
            textstr = f'Posterior means:\n'
            textstr += f'  μ = {mean_of_means:.2f}\n'
            textstr += f'  σ = {std_of_means:.2f}\n'
            textstr += f'  {within_prior:.1%} within prior ±1σ'
            
            if model_type == 'power_law':
                ax.text(-0.10, 1.05, label_list[idx], transform=ax.transAxes, fontsize=14, fontweight='bold')
            
            ax.text(0.60, 0.95, textstr, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(f'{param_labels[idx]} (Posterior Mean)', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'{param_labels[idx]}: Distribution of Posterior Means', 
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        # plt.suptitle(f'{model_type.upper()}: Control Test - Posterior Means Distribution',
        #              fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_type}_posterior_means_distribution.png',
                    dpi=300, bbox_inches='tight')
        print(f"✓ Posterior means distribution plot saved")
    
    
    
def create_control_test_table(metrics_df, models_to_test, n_sims):
    """
    Create comprehensive table for thesis/paper.
    """
    
    results = []
    
    for model_type in models_to_test:
        # Load metrics
        model_metrics_df = metrics_df[metrics_df['model_type']==model_type]
        
        param_names = ['scale1']
        if model_type == 'power_law':
            param_names.append('exponent1')
        
        for param in param_names:
            # Prior statistics (should be same for all sims)
            prior_mean = model_metrics_df[f'{param}_prior_mean'].values[0]
            prior_std = model_metrics_df[f'{param}_prior_std'].values[0]
            
            # Posterior statistics (aggregated)
            post_means = model_metrics_df[f'{param}_post_mean'].values
            post_stds = model_metrics_df[f'{param}_post_std'].values
            
            post_mean_avg = post_means.mean()
            post_mean_std = post_means.std()  # Variation across sims
            post_std_avg = post_stds.mean()
            
            # Shift metrics
            mean_shifts = model_metrics_df[f'{param}_mean_shift'].values
            relative_shifts = model_metrics_df[f'{param}_relative_shift'].values
            overlaps = model_metrics_df[f'{param}_overlap'].values
            
            # Count "successful" control (no learning)
            # Define success as: relative shift < 10% AND overlap > 80%
            success = ((np.abs(relative_shifts) < 0.10) & (overlaps > 0.80)).sum()
            success_rate = success / n_sims
            
            results.append({
                'Model': model_type,
                'Parameter': param,
                'Prior μ': f'{prior_mean:.2f}',
                'Prior σ': f'{prior_std:.2f}',
                'Posterior μ (avg)': f'{post_mean_avg:.2f} ± {post_mean_std:.2f}',
                'Posterior σ (avg)': f'{post_std_avg:.2f}',
                'Mean Shift': f'{mean_shifts.mean():.2f} ± {mean_shifts.std():.2f}',
                'Relative Shift': f'{relative_shifts.mean():.1%} ± {relative_shifts.std():.1%}',
                'Overlap': f'{overlaps.mean():.1%} ± {overlaps.std():.1%}',
                'Success Rate': f'{success_rate:.1%}'
            })
    
    table_df = pd.DataFrame(results)
    
    
    return table_df



n_sims = 100
output_dir = './output/'
models_to_test = ['exponential', 'gaussian', 'power_law']
metrics_df = pd.read_csv(f'{output_dir}prior_posterior_metrics.csv')

"summary metric"
summary_table_df = create_control_test_table(metrics_df, models_to_test, n_sims)
summary_table_df.to_csv(f'{output_dir}control_test_summary_table.csv', encoding='utf-8-sig', index=False)

"plot posterior mean"
plot_posterior_means_distribution(metrics_df, models_to_test, n_sims)

