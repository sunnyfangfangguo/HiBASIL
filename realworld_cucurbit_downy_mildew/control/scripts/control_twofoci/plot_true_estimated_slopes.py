# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 09:45:08 2025

@author: sunny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



"read trace summary"
model_types = ['exponential', 'gaussian', 'power_law']
n_sims = 19

"collect estimated parmaters"
model_type_list = []
sim_list = []
scale1_mean_list = []
scale1_hdi25_list = []
scale1_hd975_list = []

scale2_mean_list = []
scale2_hdi25_list = []
scale2_hdi975_list = []

weight_mean_list = []
weight_hdi25_list = []
weight_hdi975_list = []

exponent1_mean_list = []
exponent1_hdi25_list = []
exponent1_hdi975_list = []

exponent2_mean_list = []
exponent2_hdi25_list = []
exponent2_hdi975_list = []

for model_type in model_types:
    for sim in range(n_sims):
        trace_summary_file = f"./output/{model_type}_trace_summary_sim{sim}.csv"
        trace_summary = pd.read_csv(trace_summary_file)
        trace_summary = trace_summary.rename(columns={'Unnamed: 0':'parameter'})
        trace_summary = trace_summary.set_index('parameter')
        scale1_mean = trace_summary.at['scale1', 'mean']
        scale1_hdi25 = trace_summary.at['scale1', 'hdi_2.5%']
        scale1_hdi975 = trace_summary.at['scale1', 'hdi_97.5%']
        
        scale2_mean = trace_summary.at['scale2', 'mean']
        scale2_hdi25 = trace_summary.at['scale2', 'hdi_2.5%']
        scale2_hdi975 = trace_summary.at['scale2', 'hdi_97.5%']
        
        weight_mean = trace_summary.at['weight', 'mean']
        weight_hdi25 = trace_summary.at['weight', 'hdi_2.5%']
        weight_hdi975 = trace_summary.at['weight', 'hdi_97.5%']
        
        if model_type == "power_law":
            exponent1_mean = trace_summary.at['exponent1', 'mean']
            exponent1_hdi25 = trace_summary.at['exponent1', 'hdi_2.5%']
            exponent1_hdi975 = trace_summary.at['exponent1', 'hdi_97.5%']
            
            
            
            exponent2_mean = trace_summary.at['exponent2', 'mean']
            exponent2_hdi25 = trace_summary.at['exponent2', 'hdi_2.5%']
            exponent2_hdi975 = trace_summary.at['exponent2', 'hdi_97.5%']
            
        else:
            exponent1_mean = ""
            exponent1_hdi25 = ""
            exponent1_hdi975 = ""
            
            exponent2_mean = ""
            exponent2_hdi25 = ""
            exponent2_hdi975 = ""
            
        
        
        model_type_list.append(model_type)
        sim_list.append(sim)
        scale1_mean_list.append(scale1_mean)
        scale1_hdi25_list.append(scale1_hdi25)
        scale1_hd975_list.append(scale1_hdi975)

        scale2_mean_list.append(scale2_mean)
        scale2_hdi25_list.append(scale2_hdi25)
        scale2_hdi975_list.append(scale2_hdi975)
        
        weight_mean_list.append(weight_mean)
        exponent1_mean_list.append(exponent1_mean)
        exponent2_mean_list.append(exponent2_mean)
        
        weight_hdi25_list.append(weight_hdi25)
        weight_hdi975_list.append(weight_hdi975)

        exponent1_hdi25_list.append(exponent1_hdi25)
        exponent1_hdi975_list.append(exponent1_hdi975)

        exponent2_hdi25_list.append(exponent2_hdi25)
        exponent2_hdi975_list.append(exponent2_hdi975)
        

est_params_dict = {'model_type':model_type_list, 'simulation':sim_list, 'scale1_mean':scale1_mean_list,
                   'scale1_hdi_2.5%':scale1_hdi25_list, 'scale1_hdi_97.5%':scale1_hd975_list,
                   'scale2_mean':scale2_mean_list, 'scale2_hdi_2.5%':scale2_hdi25_list,
                   'scale2_hdi_97.5%':scale2_hdi975_list, "weight_mean":weight_mean_list, 
                   "exponent1_mean":exponent1_mean_list, "exponent2_mean":exponent2_mean_list,
                   "weight_hdi25":weight_hdi25_list, 
                   "exponent1_hdi25":exponent1_hdi25_list, "exponent2_hdi25":exponent2_hdi25_list,
                   "weight_hdi975":weight_hdi975_list, 
                   "exponent1_hdi975":exponent1_hdi975_list, "exponent2_hdi975":exponent2_hdi975_list,}

est_params_df = pd.DataFrame(data=est_params_dict)
est_params_df.to_csv("./output/estimated_parameter_mean_hdi_twofoci.csv")

# "check convergence"
# coverage_pi0 = ((est_params_df['beta_pi0_slope_hdi_2.5%'] <= 1.0) & 
#                 (est_params_df['beta_pi0_slope_hdi_97.5%'] >= 1.0)).mean() * 100

# coverage_mu = ((est_params_df['beta_mu_slope_hdi_2.5%'] <= 1.0) & 
#                (est_params_df['beta_mu_slope_hdi_97.5%'] >= 1.0)).mean() * 100

# print(f"Coverage (true=0 in 95% HDI): pi0_slope={coverage_pi0:.1f}%, mu_slope={coverage_mu:.1f}%")


# "plot estimation performace"
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# sns.boxplot(data=est_params_df, y='beta_pi0_slope_mean', hue='model_type', ax=axs[0])
# sns.boxplot(data=est_params_df, y='beta_mu_slope_mean', hue='model_type', ax=axs[1])

# plt.tight_layout()
# plt.savefig('./output/parameter_estimation_performance_Bayesian_twofocus.png', dpi=300, bbox_inches='tight')
    
        
        
        
        
