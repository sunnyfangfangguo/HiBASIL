# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:36:28 2025

@author: sunny
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



"""
logp_zero = pt.log(p_param)
logp_one = pt.log((1 - p_param) * q_param)
logp_continuous = pt.log((1 - p_param) * (1 - q_param)) + pm.Beta.logp(safe_value, alphas, betas)
here q_param is a constant value
"""



def single_focus_kernel(x, y, focus1, scale, model_type="exponential", exponent=None):
    "unpack focus parameters"
    fx, fy, fz = focus1
    
    "calculate distance"
    distance = np.sqrt((x - fx)**2 + (y - fy)**2)

    "dispersal kernel functions with shared parameters"
    if model_type == 'exponential':
        kernel = fz * np.exp(-distance / scale)
        
    elif model_type == 'gaussian':
        kernel = fz * np.exp(-(distance**2) / (2 * scale**2))
        
    elif model_type == 'power_law':
        if exponent is None:
            raise ValueError("exponent must be provided for power_law model")
            
        kernel = fz / np.power(1 + distance / scale, exponent)
        
    return kernel


def two_foci_kernel(x, y, focus1, focus2, scale1, scale2, weight, model_type="exponential", exponent1=None, exponent2=None):
    "unpack focus parameters"
    fx1, fy1, fz1 = focus1
    fx2, fy2, fz2 = focus2
    
    "calculate distance"
    distance1 = np.sqrt((x - fx1)**2 + (y - fy1)**2)
    distance2 = np.sqrt((x - fx2)**2 + (y - fy2)**2)

    "dispersal kernel functions with shared parameters"
    if model_type == 'exponential':
        kernel1 = fz1 * np.exp(-distance1 / scale1)
        kernel2 = fz2 * np.exp(-distance2 / scale2)
        
    elif model_type == 'gaussian':
        kernel1 = fz1 * np.exp(-(distance1**2) / (2 * scale1**2))
        kernel2 = fz2 * np.exp(-(distance2**2) / (2 * scale2**2))
        
    elif model_type == 'power_law':
        if (exponent1 is None) or (exponent2 is None):
            raise ValueError("exponent must be provided for power_law model")            
        kernel1 = fz1 / np.power(1 + distance1 / scale1, exponent1)
        kernel2 = fz2 / np.power(1 + distance2 / scale2, exponent2)
        
    kernel = weight * kernel1 + (1-weight) * kernel2
        
    return kernel



def sampling(sample_points, foci, prob_zero, prob_one, prob_continuous, alpha_vals, beta_vals): 
    y_observed = np.zeros(len(sample_points))
    for i in range(len(sample_points)):
        probs = np.array([prob_zero[i], prob_one[i], prob_continuous[i]])
        # Re-normalize just to be safe from minor floating point errors
        probs /= probs.sum()
        component = np.random.choice([0, 1, 2], p=probs)
        if component == 0:
            y_observed[i] = 0.0
            
        elif component == 1:
            y_observed[i] = 1.0
            
        else:
            alpha_i = alpha_vals[i]
            beta_i = beta_vals[i]
            y_observed[i] = np.random.beta(alpha_i, beta_i)
    
    y_obs = np.column_stack((sample_points, y_observed))
    y_obs_df = pd.DataFrame(y_obs, columns=['InterrowDistance', 'Distance', 'Severity'])
    y_obs_df['Plant'] = y_obs_df.index.astype(str)
    
    "add focus to observations"
    if len(foci)==3:
        fx1, fy1, fz1 = foci
        
        # 1. Calculate Euclidean distance from ALL sampled points to the focus
        distances = np.sqrt((y_obs_df['InterrowDistance'] - fx1)**2 + (y_obs_df['Distance'] - fy1)**2)
        
        # 2. Find the index of the sampled point that is CLOSEST (distance should be ~0)
        index_of_focus = distances.idxmin()
        y_obs_df.loc[index_of_focus, 'Plant'] = 'focus'
            
    elif len(foci)==6:
        fx1, fy1, fz1, fx2, fy2, fz2 = foci
        
        # Calculate distances to both foci
        distances1 = np.sqrt((y_obs_df['InterrowDistance'] - fx1)**2 + (y_obs_df['Distance'] - fy1)**2)
        distances2 = np.sqrt((y_obs_df['InterrowDistance'] - fx2)**2 + (y_obs_df['Distance'] - fy2)**2)
        
        # Find the indices to remove (the two closest points)
        index_of_focus1 = distances1.idxmin()
        index_of_focus2 = distances2.idxmin()
        
        y_obs_df.loc[index_of_focus1, 'Plant'] = 'focus'
        y_obs_df.loc[index_of_focus2, 'Plant'] = 'focus'

    y_obs_df.reset_index(drop=True, inplace=True)
       
    return y_obs_df



if __name__ == "__main__":

    sns.set_style("whitegrid")
    
    eps = 1e-08
    p0 = 0.3
    q_param = 0.01
    phi = 18
    model_type = 'power_law'
    
    # focus 1
    scale1 = 6.0
    exponent1 = 1.8
    fx1_obs = 0.0  # Focus 1 X for each observation
    fy1_obs = 5.0  # Focus 1 Y for each observation
    fz1_obs = 0.6  # Focus 1 intensity for each observation
    focus1 = [fx1_obs, fy1_obs, fz1_obs]
    weight = 0.4
    
    # focus 2
    scale2 = 5.0
    exponent2 = 1.8
    fx2_obs = 0.0  # Focus 1 X for each observation
    fy2_obs = 40.0  # Focus 1 Y for each observation
    fz2_obs = 0.8  # Focus 1 intensity for each observation
    focus2 = [fx2_obs, fy2_obs, fz2_obs]

    
    # Define the varying coordinates
    y_coords = np.linspace(start=-2, stop=50, num=250) # 10 points from Y=0 to Y=12
    
    # ensure focus location is in the coords list
    y_coords = np.unique(np.concatenate([[fy1_obs], y_coords, [fy2_obs]]))
    
    x_coords = np.full(y_coords.shape, 0.0) # X is always 0.0
    sample_points = np.column_stack((x_coords, y_coords))  # np.column_stack puts the arrays side-by-side
    
    sample_kernel = two_foci_kernel(sample_points[:,0], sample_points[:, 1], focus1, focus2, scale1, 
                                    scale2, weight, model_type=model_type, 
                                    exponent1=exponent1, exponent2=exponent2)
    
    p_param = np.clip(p0 * (1- sample_kernel), eps, 1- eps)
    mu = np.clip(sample_kernel, eps, 1- eps)
    
    alpha_samples = mu * phi
    beta_samples = (1 - mu) * phi
    
    prob_zero = p_param
    prob_one = (1.0 - p_param) * q_param
    prob_continuous = (1.0 - p_param) * (1.0 - q_param)
       
    # Create figure with contour plots for each row
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    "contour plot"
    # Calculate distances from each observation to its row's foci
    # Use advanced indexing to get row-specific focus coordinates
    X, Y = np.mgrid[-10:10:100j, -5:55:100j]
    
    # calculate kernels with shared weight
    mu_all_grid = two_foci_kernel(X, Y, focus1, focus2, scale1, 
                                    scale2, weight, model_type=model_type, 
                                    exponent1=exponent1, exponent2=exponent2)
    
    contour = axs[0].contourf(X, Y, mu_all_grid, levels=15, cmap='viridis', alpha=0.7)
    
    "scatter plot"
    y_observed = sampling(sample_points, focus1, prob_zero, prob_one, prob_continuous, alpha_samples, beta_samples)
    scatter = axs[0].scatter(sample_points[:,0], sample_points[:, 1], c=y_observed['Severity'], s=80, cmap='viridis', 
                       edgecolors='black', linewidth=0.8, zorder=5)
    sns.scatterplot(data=y_observed, x='Distance', y='Severity', ax=axs[1])
    
    fig.suptitle(f'{model_type.title()} Model', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig('simulation_dispersal.png', dpi=300, bbox_inches='tight')
    
                
    "simulate 100 datasets"
    df_id_current = 0
    rows_df = pd.DataFrame(columns=['InterrowDistance', 'Distance', 'Severity', 'Row', 'Simulation', 'Plant'])
    for run_id in range(2):
        np.random.seed(run_id)
        
        y_obs_run = sampling(sample_points, focus1, prob_zero, prob_one, prob_continuous, alpha_samples, beta_samples)
        row_val = run_id%4
        row_label = np.full(len(sample_points), row_val)
        sim_label = np.full(len(sample_points), run_id)
        y_obs_run['Row'] = row_label
        y_obs_run['Simulation'] = sim_label
                
        "add row label to the simulation for hierachical Bayesian model"
        df_id = int(run_id / 4)
        if df_id != df_id_current:
            rows_df.to_csv("simulated_disease_data_" + str(df_id_current) + ".csv")
            
            df_id_current = df_id
            rows_df = pd.DataFrame(columns=['InterrowDistance', 'Distance', 'Severity', 'Row', 'Simulation', 'Plant'])

        rows_df = pd.concat([rows_df, y_obs_run])
            
    rows_df.to_csv("simulated_disease_data_" + str(df_id_current) + ".csv")