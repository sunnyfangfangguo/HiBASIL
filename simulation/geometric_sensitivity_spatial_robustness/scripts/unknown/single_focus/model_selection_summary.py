# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 20:21:09 2025

@author: sunny
"""

import pandas as pd
import numpy as np


n_sims = 100
output_dir = "./output_all/"

model_types = ['exponential', 'gaussian', 'power_law']

sims = []
models = []
rank_loos = []
elpd_loos = []
p_loos = []
se_d_loos = []
delta_elpds = [] # differ to ELPD of the rank 0 model

for sim in range(n_sims):
    file = output_dir + "models_performances_sim" + str(sim) + ".csv"
    performance_df = pd.read_csv(file, index_col=0)
    
    for model_type in model_types:
        rank_loo = performance_df.at['rank_loo', model_type]
        elpd_loo = performance_df.at['elpd_loo', model_type]
        p_loo = performance_df.at['p_loo', model_type]
        se_d_loo = performance_df.at['se_d_loo', model_type]
        
        rank0_model = performance_df.columns[performance_df.loc['rank_loo'] == '0'].tolist()[0]
        rank0_elpd = performance_df.at['elpd_loo', rank0_model]
        delta_elpd = float(elpd_loo) - float(rank0_elpd)
        
        sims.append(sim)
        models.append(model_type)
        rank_loos.append(rank_loo)
        elpd_loos.append(elpd_loo)
        p_loos.append(p_loo)
        se_d_loos.append(se_d_loo)
        delta_elpds.append(delta_elpd)
        
model_select_df = pd.DataFrame({'Simulation':sims, 'Model':models, 'rank_loo':rank_loos,
                                'elpd_loo':elpd_loos, 'p_loo':p_loos, 'se_d_loo':se_d_loos,
                                'delta_elpd':delta_elpds}) 

# delta ELPD below the threshold of significance (< 2* SE), effectively treating models as predictive equivalents 
model_select_df['Predictive Equivalent'] = (abs(model_select_df['delta_elpd']) <= 2 * model_select_df['se_d_loo'].astype(float))

model_select_df.to_csv(output_dir + "model_selection_summary.csv")       
