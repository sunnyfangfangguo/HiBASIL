# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 19:23:16 2025

@author: sunny
"""

"comple null, one focus, two foci models' elpd, r2....."

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm



def model_performance(traces):
    """
    model selection based on loo, waic
    """    
    "compare with loo"
    loo_comparison = az.compare(traces,ic='loo')
    
    comparisons = []
    for model_name, trace in traces.items():
        # effective number of parameters
        waic_result = az.waic(trace)
        p_waic = float(waic_result.p_waic)
        n_obs = int(waic_result.n_data_points)
               
        row = loo_comparison.loc[model_name]
        elpd_loo = row['elpd_loo'] # Like 'log_likelihood' but better
        p_loo = row['p_loo'] # Effective parameters 
        rank_loo = row['rank']  # 0=best, 1=second, etc.
        se_d_loo = row['dse'] # Standard error of difference
        warning = row['warning'] # Any computational warnings
        
        model_loo = az.loo(traces[model_name])
        pareto_k_bad = int(np.sum(model_loo.pareto_k > 0.7)) # Problematic observations
                                
        comparisons.append({'model':model_name, 'n_obs':n_obs, 
                               'elpd_loo': elpd_loo, 
                               'p_loo': p_loo, 'rank_loo':rank_loo,
                               'se_d_loo':se_d_loo, 'warning':warning, 
                               'pareto_k_bad':pareto_k_bad})
        
    comparisons_df = pd.DataFrame(comparisons)
    
    return comparisons_df



"compare models - traces files"
traces = {}
# null model
null_trace_file = "./nullmodel/output/null_model_trace.nc"
null_trace = az.from_netcdf(null_trace_file)
traces['null_model'] = null_trace

# one focus model
onefocus_exponential_trace_file = "./singlefocus_model/output/exponential_trace.nc"
onefocus_exponential_trace = az.from_netcdf(onefocus_exponential_trace_file)
traces['onefocus_exponential'] = onefocus_exponential_trace 

onefocus_gaussian_trace_file = "./singlefocus_model/output/gaussian_trace.nc"
onefocus_gaussian_trace = az.from_netcdf(onefocus_gaussian_trace_file)
traces['onefocus_gaussian'] = onefocus_gaussian_trace

onefocus_powerlaw_trace_file = "./singlefocus_model/output/power_law_trace.nc"
onefocus_powerlaw_trace = az.from_netcdf(onefocus_powerlaw_trace_file)
traces['onefocus_power_law'] = onefocus_powerlaw_trace

# two foci model
twofoci_exponential_trace_file = "./twofoci_model/output/exponential_trace.nc"
twofoci_exponential_trace = az.from_netcdf(twofoci_exponential_trace_file)
traces['twofoci_exponential'] = twofoci_exponential_trace

twofoci_gaussian_trace_file = "./twofoci_model/output/gaussian_trace.nc"
twofoci_gaussian_trace = az.from_netcdf(twofoci_gaussian_trace_file)
traces['twofoci_gaussian'] = twofoci_gaussian_trace

twofoci_powerlaw_trace_file = "./twofoci_model/output/power_law_trace.nc"
twofoci_powerlaw_trace = az.from_netcdf(twofoci_powerlaw_trace_file)
traces['twofoci_power_law'] = twofoci_powerlaw_trace



"posterior predictive check files"
null_ppc_file = "./nullmodel/output/ppc_metrics_all_sims.csv"
null_ppc_df = pd.read_csv(null_ppc_file)
null_ppc_df['model'] = 'null_model'

onefocus_ppc_file = "./singlefocus_model/output/ppc_metrics_all_sims.csv"
onefocus_ppc_df = pd.read_csv(onefocus_ppc_file)
onefocus_ppc_df['model'] = 'onefocus_' + onefocus_ppc_df['model_type']

twofoci_ppc_file = "./twofoci_model/output/ppc_metrics_all_sims.csv"
twofoci_ppc_df = pd.read_csv(twofoci_ppc_file)
twofoci_ppc_df['model'] = 'twofoci_' + twofoci_ppc_df['model_type']

ppc_df = pd.concat([null_ppc_df, onefocus_ppc_df, twofoci_ppc_df], ignore_index=True)

comparisons_df = model_performance(traces)

combined_df = pd.merge(
    ppc_df, 
    comparisons_df, 
    on='model', 
    how='outer'
)

combined_df.to_csv("snow_models_performace_comparison.csv")   
