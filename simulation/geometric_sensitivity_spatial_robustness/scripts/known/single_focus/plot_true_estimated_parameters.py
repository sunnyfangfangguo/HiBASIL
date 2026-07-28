# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 09:45:08 2025

@author: sunny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob


"read trace summary"
model_types = ['exponential', 'gaussian', 'power_law']
n_sims = 100
output_dir = './output/'



def extract_metrics(df, sim, model_type, true_values_dict):
    metrics = {'simulation':sim, 'model':model_type}
    
    "global parameters"
    if model_type == 'power_law':
        global_params = ['scale1', 'exponent1', 'q_param', 'pi0', 'phi']
        
    elif model_type in ['exponential', 'gaussian']:
        global_params = ['scale1', 'q_param', 'pi0', 'phi']
        
    for param in global_params:
        row = df.loc[param]
        true_val = true_values_dict.get(param)
        
        # store posterior estimates
        metrics[f'{param}_post_mean'] =  row['mean']
        metrics[f'{param}_rhat'] = row['r_hat']
        metrics[f'{param}_ess_bulk'] = row['ess_bulk']
        
        # calculate accuracy metrics
        bias = row['mean'] - true_val
        relative_err = abs(bias) / true_val * 100 if true_val != 0 else 0
        covered = row['hdi_2.5%'] <= true_val <= row['hdi_97.5%']
        
        metrics[f'{param}_bias'] = bias
        metrics[f'{param}_abs_bias'] = abs(bias)
        metrics[f'{param}_relative_error'] = relative_err
        metrics[f'{param}_covered'] = 1 if covered else 0
        metrics[f'{param}_converged'] = 1 if row['r_hat'] <= 1.01 else 0
        
        
    "mu parameters"
    mu_rows = df[df.index.str.startswith('mu[', na=False)]    
    mu_true = true_values_dict.get('mu')
    
    post_means = mu_rows['mean'].values
    hdis_lower = mu_rows['hdi_2.5%'].values
    hdis_upper = mu_rows['hdi_97.5%'].values

    biases = post_means - mu_true
    coverage = ((hdis_lower <= mu_true) & (mu_true <= hdis_upper)).mean()
    
    # store mu posterior estimates
    metrics['n_mu'] = len(mu_rows)
    metrics['mu_post_mean_avg'] = post_means.mean()
    metrics['mu_post_mean_sd'] = post_means.std()
    metrics['mu_post_mean_min'] = post_means.min()
    metrics['mu_post_mean_max'] = post_means.max()
    
    metrics['mu_bias_mean'] = biases.mean()
    metrics['mu_bias_sd'] = biases.std()
    metrics['mu_rmse'] = np.sqrt((biases**2).mean())
    metrics['mu_mae'] = np.abs(biases).mean()
    
    metrics['mu_coverage_rate'] = coverage
    metrics['mu_n_covered'] = ((hdis_lower <= mu_true) & (mu_true <= hdis_upper)).sum()
    
    # convergence
    rhat_valid = mu_rows['r_hat'].dropna()
    metrics['mu_n_rhat_nan'] = mu_rows['r_hat'].isna().sum()
    
    if len(rhat_valid) > 0:
        metrics['mu_rhat_mean'] = rhat_valid.mean()
        metrics['mu_rhat_max'] = rhat_valid.max()
        metrics['mu_n_rhat_above_1.01'] = (rhat_valid > 1.01).sum()
        
    else:
        metrics['mu_rhat_mean'] = np.nan
        metrics['mu_rhat_max'] = np.nan
        metrics['mu_n_rhat_above_1.01'] = 0
        
    metrics['mu_ess_bulk_min'] = mu_rows['ess_bulk'].min()
    metrics['mu_ess_bulk_mean'] = mu_rows['ess_bulk'].mean()
    metrics['mu_n_ess_below_400'] = (mu_rows['ess_bulk'] < 400).sum()
    
    return metrics
        

def aggregate_across_sims(n_sims, model_type, true_values_dict):
    
    all_metrics = []
    
    for sim in range(n_sims):
        trace_summary_file = f"{output_dir}{model_type}_trace_summary_sim{sim}.csv"
            
        df = pd.read_csv(trace_summary_file, index_col=0)
        metrics = extract_metrics(df, sim, model_type, true_values_dict)
        
        all_metrics.append(metrics)
        
    return pd.DataFrame(all_metrics)


def create_summary_table_global(all_metrics_df, model_type, true_values_dict):
    "global parameters"
    if model_type == 'power_law':
        global_params = ['scale1', 'exponent1', 'q_param', 'pi0', 'phi']
        
    elif model_type in ['exponential', 'gaussian']:
        global_params = ['scale1', 'q_param', 'pi0', 'phi']
    
    n_sims = len(all_metrics_df)            
    
    summaries = {}
    statistic_data = []
    for param in global_params:
        # true value
        true_val = true_values_dict.get(param)
        
        # convergence statistics
        n_converged = all_metrics_df[f'{param}_converged'].sum()
        n_low_ess = (all_metrics_df[f'{param}_ess_bulk'] < 400).sum()
        
        # create statistics dict
        param_summary = {
            'Model':model_type,
            'Parameter':param,
            'True Value':f'{true_val:.2f}',
            'N Simulations':f'{n_sims}',
            'Mean Estimate':f"{all_metrics_df[f'{param}_post_mean'].mean():.2f} ± {all_metrics_df[f'{param}_post_mean'].std():.2f}",
            'Mean Bias':f"{all_metrics_df[f'{param}_bias'].mean():.2f} ± {all_metrics_df[f'{param}_bias'].std():.2f}",
            'Mean Abs Bias':f"{all_metrics_df[f'{param}_abs_bias'].mean():.2f}",
            'Mean Relative Error':f"{all_metrics_df[f'{param}_relative_error'].mean():.2f}",
            'Coverage':f"{all_metrics_df[f'{param}_covered'].mean()*100:.1f}%",
            'Converged':f"{(n_converged/n_sims)*100:.1f}%",
            'Low ESS':f"{(n_low_ess/n_sims)*100:.1f}%",
            'Mean R-hat':f"{all_metrics_df[f'{param}_rhat'].mean():.2f}",
            'Mean ESS (bulk)':f"{all_metrics_df[f'{param}_ess_bulk'].mean():.0f}"
            }
        
        statistic_data.append(param_summary)
   
    return pd.DataFrame(statistic_data)
        
        

def create_summary_table_mu(all_metrics_df, model_type, true_values_dict):
    
    true_val = true_values_dict.get('mu')
    n_sims = len(all_metrics_df)
    
    data = {
        'Metric':[],
        'Value':[]
        }
    
    data['Metric'].append('Model')
    data['Value'].append(model_type)
    
    data['Metric'].append('True Value')
    data['Value'].append(f"{true_val:.2f}")
    
    data['Metric'].append('N Simulations')
    data['Value'].append(f"{n_sims}")
    
    data['Metric'].append('Mean Estimate')
    data['Value'].append(f"{all_metrics_df['mu_post_mean_avg'].mean():.2f} ± {all_metrics_df['mu_post_mean_avg'].std():.2f}")
    
    data['Metric'].append('Mean Bias')
    data['Value'].append(f"{all_metrics_df['mu_bias_mean'].mean():.2f} ± {all_metrics_df['mu_bias_mean'].std():.2f}")
    
    data['Metric'].append('RMSE')
    data['Value'].append(f"{all_metrics_df['mu_rmse'].mean():.2f} ± {all_metrics_df['mu_rmse'].std():.2f}")
    
    data['Metric'].append('Coverage Rate')
    data['Value'].append(f"{all_metrics_df['mu_coverage_rate'].mean()*100:.2f} ± {all_metrics_df['mu_coverage_rate'].std()*100:.2f}")
    
    "convergence metrics"
    if not all_metrics_df['mu_rhat_mean'].isna().all():
        data['Metric'].append('Mean R-hat')
        data['Value'].append(f"{all_metrics_df['mu_rhat_mean'].mean():.2f}")
    else:
        data['Metric'].append('R-hat Status')
        data['Value'].append('All NaN')
        
    data['Metric'].append('Mean ESS (bulk)')
    data['Value'].append(f"{all_metrics_df['mu_ess_bulk_mean'].mean():.0f}")
    
    data['Metric'].append('N with ESS < 400')
    data['Value'].append(f"{all_metrics_df['mu_n_ess_below_400'].mean():.1f}")
    
    
    return pd.DataFrame(data)
        
    
        

def create_summary_table_pparam(all_metrics_df, model_type, true_values_dict):
    
    true_val = true_values_dict.get('p_param')
    n_sims = len(all_metrics_df)
    
    data = {
        'Metric':[],
        'Value':[]
        }
    
    data['Metric'].append('Model')
    data['Value'].append(model_type)
    
    data['Metric'].append('True Value')
    data['Value'].append(f"{true_val:.2f}")
    
    data['Metric'].append('N Simulations')
    data['Value'].append(f"{n_sims}")
    
    data['Metric'].append('Mean Estimate')
    data['Value'].append(f"{all_metrics_df['p_param_post_mean_avg'].mean():.2f} ± {all_metrics_df['p_param_post_mean_avg'].std():.2f}")
    
    data['Metric'].append('Mean Bias')
    data['Value'].append(f"{all_metrics_df['p_param_bias_mean'].mean():.2f} ± {all_metrics_df['p_param_bias_mean'].std():.2f}")
    
    data['Metric'].append('RMSE')
    data['Value'].append(f"{all_metrics_df['p_param_rmse'].mean():.2f} ± {all_metrics_df['p_param_rmse'].std():.2f}")
    
    data['Metric'].append('Coverage Rate')
    data['Value'].append(f"{all_metrics_df['p_param_coverage_rate'].mean()*100:.2f} ± {all_metrics_df['p_param_coverage_rate'].std()*100:.2f}")
    
    "convergence metrics"
    if not all_metrics_df['mu_rhat_mean'].isna().all():
        data['Metric'].append('Mean R-hat')
        data['Value'].append(f"{all_metrics_df['p_param_rhat_mean'].mean():.2f}")
    else:
        data['Metric'].append('R-hat Status')
        data['Value'].append('All NaN')
        
    data['Metric'].append('Mean ESS (bulk)')
    data['Value'].append(f"{all_metrics_df['p_param_ess_bulk_mean'].mean():.0f}")
    
    data['Metric'].append('N with ESS < 400')
    data['Value'].append(f"{all_metrics_df['p_param_n_ess_below_400'].mean():.1f}")
    
    
    return pd.DataFrame(data)
        
    
    
for model_type in model_types:

    if model_type == "power_law":
        true_values_dict = {'scale1':5.0, 'exponent1':2.0, 'q_param':0.01, 'pi0':0.3, 
                        'phi':18, 'p_param':0.3, 'mu':0.2}
    elif model_type in ['exponential', 'gaussian']:
        true_values_dict = {'scale1':5.0, 'q_param':0.01, 'pi0':0.3, 
                        'phi':18, 'p_param':0.3, 'mu':0.2}
        
    
    all_metrics_df = aggregate_across_sims(n_sims, model_type, true_values_dict)    
    all_metrics_df.to_csv(f"{output_dir}{model_type}_all_metrics_df_summary.csv")        
    
    global_params_summary = create_summary_table_global(all_metrics_df, model_type, true_values_dict)        

    
    global_params_summary.to_csv(f"{output_dir}{model_type}_global_params_summary.csv", encoding='utf-8-sig')




