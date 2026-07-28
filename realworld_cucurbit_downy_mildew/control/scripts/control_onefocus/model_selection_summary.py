# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 20:21:09 2025

@author: sunny
"""

import pandas as pd
import numpy as np


n_sims = 87
output_dir = "./output/"

model_types = ['exponential', 'gaussian', 'power_law']

sims = []
models = []
rank_loos = []
elpd_loos = []
p_loos = []

for sim in range(n_sims):
    file = output_dir + "models_performances_sim" + str(sim) + ".csv"
    performance_df = pd.read_csv(file, index_col=0)
    
    for model_type in model_types:
        rank_loo = performance_df.at['rank_loo', model_type]
        elpd_loo = performance_df.at['elpd_loo', model_type]
        p_loo = performance_df.at['p_loo', model_type]
        
        sims.append(sim)
        models.append(model_type)
        rank_loos.append(rank_loo)
        elpd_loos.append(elpd_loo)
        p_loos.append(p_loo)
        
model_select_df = pd.DataFrame({'Simulation':sims, 'Model':models, 'rank_loo':rank_loos,
                                'elpd_loo':elpd_loos, 'p_loo':p_loos}) 
model_select_df.to_csv("model_selection_summary.csv")       
