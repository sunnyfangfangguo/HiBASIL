# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:10:32 2025

@author: sunny
"""

# -*- coding: utf-8 -*-
"""
Multi-Row ZOIB Bayesian Model for Disease Severity Analysis
Fits all rows together with shared dispersal parameters but row-specific foci
"""

import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import arviz as az
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
import warnings
import matplotlib.patches as mpatches


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

"Because the true slope is 0, 0 * kernel will multiply out kernel. The kernel parameters should be the true value. "

eps = 1e-12

def quality_control_check(df):
    """
    Remove measurement errors and document exclusions
    """
    "fill NaN values with row means for the foci"
    focusnan_mask = (df['Plant'] == 'focus') & (df['Severity_Count'].isna())

    if (focusnan_mask.any()):
        row_means = df.groupby('Row')['Severity_Count'].mean()
        df.loc[focusnan_mask, 'Severity_Count'] = df.loc[focusnan_mask, 'Row'].map(row_means)
        
    "drop nan value in general plants"
    df = df.dropna(subset=['Severity_Count'])

    return df


def severity2proportion(x):
    """Convert severity to proportion [0,1]"""
    if pd.isna(x):
        return np.nan
    return np.clip(x / 100, 0.0, 1.0)


def load_and_process_multi_row_data(df):
    """Load and process the complete multi-row dataset"""    
    # calculate severity proportion
    df['Severity'] = df['Severity_Count'].apply(severity2proportion)  
    
    # Separate foci from observations
    foci_data = df[df['Plant'] == 'focus'].copy()
    # obs_data = df[df['Plant'] != 'focus'].copy()
    obs_data = df.copy()
    
    print(f"Foci locations: {len(foci_data)} points")
    print(f"Observations: {len(obs_data)} points")
    print(f"Unique rows: {sorted(obs_data['Row'].unique())}")
    
    # Extract focus coordinates for each row
    focus_coords = {}
    for row_id in obs_data['Row'].unique():
        row_foci = foci_data[foci_data['Row'] == row_id]
        if len(row_foci) == 1:
            # Sort by Distance to get consistent focus1 (Y=0) and focus2 (Y=~100)
            row_foci_sorted = row_foci.sort_values('Distance')
            focus1 = row_foci_sorted.iloc[0]
            
            focus_coords[row_id] = {
                'focus1': (focus1['InterrowDistance'], focus1['Distance'], focus1['Severity']),
            }
            
            print(f"Row {row_id}: Focus1=({focus1['InterrowDistance']}, {focus1['Distance']}, {focus1['Severity']:.1f})")
    
    return obs_data, focus_coords



def zoib_logpdf(value, p_param, q_param, alphas, betas):
    """ZOIB log-likelihood function"""
    
    # Clip parameters
    p_param = pt.clip(p_param, eps, 1 - eps)
    q_param = pt.clip(q_param, eps, 1 - eps)
    alphas = pt.clip(alphas, eps, 1000)
    betas = pt.clip(betas, eps, 1000)
    
    # Masks
    is_zero = pt.eq(value, 0.0)
    is_one = pt.eq(value, 1.0)
    
    # Safe masking for Beta evaluation
    safe_value = pt.where(is_zero, 0.5, pt.where(is_one, 0.5, value))
    
    # Components
    logp_zero = pt.log(p_param)
    logp_one = pt.log((1 - p_param) * q_param)
    logp_continuous = pt.log((1 - p_param) * (1 - q_param)) + pm.Beta.logp(safe_value, alphas, betas)
    
    # Clean selection
    result = pt.switch(is_zero, logp_zero, pt.switch(is_one, logp_one, logp_continuous))
     
    return result



def zoib_random_fn(p_param, q_param, alpha_vals, beta_vals, rng=None, size=None):
    """
    Random number generator for ZOIB using categorical component selection.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Ensure inputs are arrays
    p_param = np.asarray(p_param)
    q_param = np.asarray(q_param)
    alpha_vals = np.asarray(alpha_vals)
    beta_vals = np.asarray(beta_vals)
    
    # Get output shape from spatially-varying parameters
    output_shape = alpha_vals.shape
    n_obs = np.prod(output_shape)
       
    # Calculate component probabilities for ZOIB
    prob_zero = p_param
    prob_one = (1 - p_param) * q_param
    prob_beta = (1 - p_param) * (1 - q_param)
    
    # Stack probabilities for categorical choice (shape: [n_obs, 3])
    probs = np.stack([prob_zero.flatten(), 
                      prob_one.flatten(), 
                      prob_beta.flatten()], axis=-1)
    
    # Draw component membership for each observation
    components = np.array([rng.choice(3, p=probs[i]) for i in range(n_obs)])
    components = components.reshape(output_shape)
    
    # Initialize output
    out = np.zeros(output_shape)
    
    # Fill in values based on component
    zero_mask = components == 0
    one_mask = components == 1
    beta_mask = components == 2
    
    out[zero_mask] = 0.0
    out[one_mask] = 1.0
    
    # Generate beta values for continuous component
    if np.any(beta_mask):
        out[beta_mask] = rng.beta(alpha_vals[beta_mask], beta_vals[beta_mask])
    
    return out



def create_multi_row_zoib_model(obs_data, focus_coords, model_type='exponential'):
    """
    Create ZOIB model using all rows with shared dispersal parameters
    """

    # Prepare data arrays
    X_data = np.array(obs_data['InterrowDistance'].values, dtype=np.float64)
    Y_data = np.array(obs_data['Distance'].values, dtype=np.float64)
    severity_data = np.array(obs_data['Severity'].values, dtype=np.float64)
    row_ids = np.array(obs_data['Row'].values, dtype=int)
    
    n_obs = len(severity_data)
    unique_rows = sorted(obs_data['Row'].unique())
    n_rows = len(unique_rows)
    row_to_idx = {row: idx for idx, row in enumerate(unique_rows)}
    row_indices = np.array([row_to_idx[row] for row in row_ids])
    
    print("\nMulti-row model preparation:")
    print(f"- Total observations: {n_obs}")
    print(f"- Number of rows: {n_rows}")
    print(f"- Rows: {unique_rows}")
    print(f"- Severity range: [{severity_data.min():.1f}, {severity_data.max():.1f}]")
    
    # Check for edge cases
    n_zeros = np.sum(severity_data == 0.0)
    n_ones = np.sum(severity_data == 1.0)
    n_continuous = np.sum((severity_data > 0) & (severity_data < 1))
    
    print(f"- Exact zeros: {n_zeros} ({n_zeros/n_obs*100:.1f}%)")
    print(f"- Exact ones: {n_ones} ({n_ones/n_obs*100:.1f}%)")
    print(f"- Continuous: {n_continuous} ({n_continuous/n_obs*100:.1f}%)")
    
    # Extract focus coordinates for each row
    focus1_coords = np.zeros((n_rows, 3))
    
    for idx, row in enumerate(unique_rows):
        coords = focus_coords[row]
        focus1_coords[idx] = [coords['focus1'][0], coords['focus1'][1], coords['focus1'][2]]
                 
    with pm.Model() as model:
        # Convert to tensor variables
        X = pt.as_tensor_variable(X_data)
        Y = pt.as_tensor_variable(Y_data)
        row_idx = pt.as_tensor_variable(row_indices)
        
        # Known focus coordinates (fixed, not estimated)
        fx1 = pt.as_tensor_variable(focus1_coords[:, 0])  # Shape: (n_rows,)
        fy1 = pt.as_tensor_variable(focus1_coords[:, 1])
        
        # Store data for later use
        pm.Data("X_data", X_data)
        pm.Data("Y_data", Y_data)
        pm.Data("severity_data", severity_data)
        pm.Data("row_indices", row_indices)
        
        if model_type == 'exponential':
            scale1 = pm.Gamma("scale1", alpha=2, beta=2)  
            
        elif model_type == 'gaussian': 
            scale1 = pm.Gamma("scale1", alpha=2, beta=2) 
            
        elif model_type == 'power_law':
            scale1 = pm.Gamma("scale1", alpha=2, beta=2) 
            exponent1 = pm.Gamma("exponent1", alpha=15, beta=2.0) 
         
        # Shared dispersal parameters (same across all rows)
        fz1_base = pm.Beta("fz1_base", alpha=1, beta=1)  # Base intensity for focus 1
        
        # Row-specific random effects on intensity
        sigma_row = pm.HalfNormal("sigma_row", sigma=0.2)
        fz1_row_effect = pm.Normal("fz1_row_effect", mu=0, sigma=sigma_row, shape=n_rows)
        
        # Row-specific intensities
        fz1 = pm.Deterministic("fz1", pt.clip(fz1_base + fz1_row_effect, 0.001, 0.99))
        
        # ZOIB mixture parameters (shared across rows)
        q_param = pm.Beta("q_param", alpha=1, beta=10)  # Probability of exact 1 given not 0, constant in space
        pi0 = pm.Beta("pi0", alpha=2, beta=5)
        
        # Precision parameter for Beta distribution
        phi = pm.Gamma("phi", alpha=4, beta=2) # constant in space
        
        # Calculate distances from each observation to its row's foci
        # Use advanced indexing to get row-specific focus coordinates
        fx1_row = fx1[row_idx]  # Focus 1 X for each row
        fy1_row = fy1[row_idx]  # Focus 1 Y for each row
        fz1_row = fz1[row_idx]  # Focus 1 intensity for each row
        
        distance = pt.sqrt((X - fx1_row)**2 + (Y - fy1_row)**2)
        
        "Kernel functions with shared parameters"
        if model_type == 'exponential':
            kernel = fz1_row * pt.exp(-distance / pt.maximum(scale1, 0.01))
            
        elif model_type == 'gaussian':
            kernel = fz1_row * pt.exp(-(distance**2) / pt.maximum(2 * scale1**2, 0.01))
            
        elif model_type == 'power_law':
            kernel = fz1_row / pt.pow(1 + distance / pt.maximum(scale1, 0.01), exponent1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
                
        mu = pm.Deterministic("mu", pt.clip(kernel, eps, 1-eps))    
        p_param = pm.Deterministic("p_param", pt.clip(pi0 * (1 - kernel), eps, 1-eps))
        
        # Beta distribution parameters
        alpha_vals = mu * phi
        beta_vals = (1 - mu) * phi
                
        # Create observation model
        obs = pm.CustomDist(
            "obs", 
            p_param, q_param, alpha_vals, beta_vals,
            logp=zoib_logpdf,
            random=zoib_random_fn,
            observed=severity_data,
            shape=n_obs
        )
    
    return model



def convergence_diagnostics(trace):
    """
    Extract R-hat and divergence statistics from MCMC trace
    """
    diagnostics = {
        'max_rhat': None,
        'n_divergent': None,
        'rhat_converged': False,
        'divergence_acceptable': False,
        'rhat_error': None,
        'divergence_error': None
    }
    
    "R-hat calculation"    
    r_hat = az.rhat(trace)
    
    if hasattr(r_hat, 'to_array'):
        r_hat_array = r_hat.to_array()
        max_r_hat = float(r_hat_array.max().values)
    elif hasattr(r_hat, 'max'):
        max_val = r_hat.max()
        max_r_hat = float(max_val.values if hasattr(max_val, 'values') else max_val)
    else:
        max_values = []
        for var_name in r_hat.data_vars:
            var_data = r_hat[var_name]
            if hasattr(var_data, 'values'):
                max_values.append(float(var_data.values.max()))
            else:
                max_values.append(float(var_data.max()))
        max_r_hat = max(max_values)
    
    diagnostics['max_rhat'] = max_r_hat
    diagnostics['rhat_converged'] = max_r_hat <= 1.1
            
    "divergence calculation"    
    diverging = trace.sample_stats.diverging
    if hasattr(diverging, 'sum'):
        if hasattr(diverging.sum(), 'values'):
            n_divergent = int(diverging.sum().values)
        else:
            n_divergent = int(diverging.sum())
    else:
        n_divergent = int(diverging)
    
    diagnostics['n_divergent'] = n_divergent
    diagnostics['divergence_acceptable'] = n_divergent < 10
    
    diagnostics_df = pd.DataFrame(data=diagnostics, index=[0])
    
    
    "Basic summary"
    summary = az.summary(trace, hdi_prob=0.95)
    print("✓ Parameter summary computed")
    print(f"Parameters estimated: {len(summary)}")

    # Print parameter estimates
    print("\nShared dispersal parameters (mean ± std):")
    shared_params = ['fz1_base', 'scale1', 'p', 'q', 'phi']
    for param in shared_params:
        if param in summary.index:
            mean_val = summary.loc[param, 'mean']
            std_val = summary.loc[param, 'sd']
            print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Print row effects
    print("\nRow-specific effects:")
    row_params = [p for p in summary.index if 'row_effect' in p or 'sigma_row' in p]
    for param in row_params:
        mean_val = summary.loc[param, 'mean']
        std_val = summary.loc[param, 'sd']
        print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")
                        
    return diagnostics_df, summary



def calculate_all_ess(trace):
    """Calculate ESS for all parameters and return as DataFrame"""
    
    param_names = list(trace.posterior.data_vars.keys())    
    results = []    
    for param in param_names:
        param_data = trace.posterior[param]
        
        if param_data.ndim == 2:  # Scalar parameter (chains, draws)            
            ess_value = az.ess(param_data, method='tail').values
            results.append({
                'parameter': param,
                'element': None,
                'ess': ess_value
            })

                
        elif param_data.ndim == 3:  # Array parameter (chains, draws, dim)
            dim_size = param_data.shape[2]
            for i in range(dim_size):                
                param_element = param_data.isel({f'{param}_dim_0': i})
                ess_value = az.ess(param_element, method='tail').values
                results.append({
                    'parameter': param,
                    'element': i,
                    'ess': ess_value
                })
    
    return pd.DataFrame(results)



def plot_multi_row_results(trace, obs_data, focus_coords, y_pred, y_low, y_high, model_type):
    """Create visualization of the multi-row results"""    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    plot_labels = ['A', 'B', 'C']
    
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']

    # Get row effects
    unique_rows = sorted(obs_data['Row'].unique())
    
    # Create subplots for each row
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 6))
    axes = axes.flatten()
    
    for idx, row in enumerate(unique_rows):
        ax = axes[idx]
        row_mask = (obs_data['Row'] == row)
        
        # Get data for this row
        row_data = obs_data[obs_data['Row'] == row]
        # row_data = row_data.sort_values(by=['Distance', 'InterrowDistance'])
        row_x = row_data['InterrowDistance'].values
        row_y = row_data['Distance'].values
        row_sev = row_data['Severity'].values
        
        # Get focus coordinates for this row
        focus1_coords = focus_coords[row]['focus1']
        
        # Row-specific intensities
        fx1_row = focus1_coords[0]
        fy1_row = focus1_coords[1]
        fz1_row = summary.loc[f'fz1[{idx}]', 'mean']
        
        # Create prediction function for this row
        def predict_row(x, y):
            distance = np.sqrt((x - fx1_row)**2 + (y - fy1_row)**2)
            
            if model_type == 'exponential':
                kernel = fz1_row * np.exp(-distance / scale1_est)
            
            elif model_type == 'gaussian':
                kernel = fz1_row * np.exp(-(distance**2) / (2 * scale1_est**2))
                
            elif model_type == 'power_law':
                kernel = fz1_row / np.power(1 + distance / scale1_est, exponent1_est)
                            
            return kernel
        

        "calculate R2 from posterior predictive"
        y_pred_row = y_pred[row_mask.values]
        y_low_row = y_low[row_mask.values]
        y_high_row = y_high[row_mask.values]
        r2 = r2_score(row_sev, y_pred_row) # y_pred is from posterior predicitve
        
        # Get local sorting index for the plot line
        sort_idx = np.argsort(row_y)
        
        # Plot observed vs predicted
        ax.scatter(row_y, row_sev, label='Observed', color='black', s=10, alpha=0.5)
        ax.fill_between(row_y[sort_idx], y_low_row[sort_idx], y_high_row[sort_idx], color='skyblue', alpha=0.3, label='95% CI')
        ax.plot(row_y[sort_idx], y_pred_row[sort_idx], label='Posterior Mean', color='navy', linewidth=2)
        
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        subplot_label = plot_labels[idx]
        ax.text(-0.05, 1.05, subplot_label, transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top')
        
        # Mark foci
        ax.axvline(focus1_coords[1], color='black', linestyle='--', alpha=0.7, 
                  label=f'Observed Focus (Y={focus1_coords[1]})')
        
        ax.set_title(f'Row {row}')
        ax.set_xlabel('Distance (y)')
        ax.set_ylabel('Infection Severity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir + f'{model_type}_multi_row_results.png', dpi=600, bbox_inches='tight')
    



def plot_3d_rows_subplots(trace, obs_data, focus_coords, y_pred, model_type):
    """Create 2x2 grid of 3D subplots, one for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']
    
    # Get row effects
    unique_rows = sorted(obs_data['Row'].unique())

    # Create figure with 2x2 3D subplots
    fig = plt.figure(figsize=(16, 12))
    
    for idx, row in enumerate(unique_rows):
        # Create 3D subplot
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        # Get row-specific data
        row_data = obs_data[obs_data['Row'] == row]
        row_x = row_data['InterrowDistance'].values
        row_y = row_data['Distance'].values
        row_sev = row_data['Severity'].values
        
        # Get focus coordinates for this row
        focus1_coords = focus_coords[row]['focus1']
        
        # Row-specific intensities
        fx1_row = focus1_coords[0]
        fy1_row = focus1_coords[1]
        fz1_row = summary.loc[f'fz1[{idx}]', 'mean']
        
        # Create grid for this row's area
        row_x_min, row_x_max = row_x.min() - 1, row_x.max() + 1
        row_y_min, row_y_max = row_y.min() - 5, row_y.max() + 5
        
        X_row, Y_row = np.meshgrid(
            np.linspace(-15, 15, 50),
            np.linspace(row_y_min-5, row_y_max+5, 50)
        )
        
        # Row-specific prediction function
        def predict_row(X, Y):
            distance = np.sqrt((X - fx1_row)**2 + (Y - fy1_row)**2)
            
            if model_type == 'exponential':
                kernel = fz1_row * np.exp(-distance / scale1_est)

            elif model_type == 'gaussian':
                kernel = fz1_row * np.exp(-(distance**2) / (2 * scale1_est**2))

            elif model_type == 'power_law':
                kernel = fz1_row / np.power(1 + distance / scale1_est, exponent1_est)
            
            return kernel
        
        # Plot row-specific surface
        Z_row = predict_row(X_row, Y_row)
        surf = ax.plot_surface(X_row, Y_row, Z_row, 
                             cmap='viridis', alpha=0.7, linewidth=0)
        
        # Plot observed data points
        scatter = ax.scatter(row_x, row_y, row_sev, 
                           c=row_sev, s=60, cmap='viridis',
                           edgecolors='black', linewidth=0.5)
        
        # Mark foci with stars and vertical lines
        # Calculate predicted severity at focus locations using the model
        focus1_pred = predict_row(fx1_row, fy1_row)   
        
        ax.scatter([fx1_row], [fy1_row], [focus1_pred], 
                  marker='*', s=150, color='red', 
                  edgecolors='darkred', linewidth=2, label='Focus 1')
        ax.plot([fx1_row, fx1_row], 
               [fy1_row, fy1_row], 
               [0, focus1_pred], 'r-', linewidth=3, alpha=0.8)
        
        "calculate R2 from posterior predictive"
        row_mask = obs_data['Row'] == row
        y_pred_row = y_pred[row_mask.values]
        r2_row = r2_score(row_sev, y_pred_row)
        
        # Set labels and title
        ax.set_xlabel('InterrowDistance', fontsize=10)
        ax.set_ylabel('Distance', fontsize=10)
        ax.set_zlabel('Severity', fontsize=10)
        ax.set_title(f'Row {row}: R² = {r2_row:.3f}\nfz={fz1_row:.3f}', fontsize=12)
        
        # Set consistent viewing angle
        ax.view_init(elev=45, azim=45)
        
        # Only show legend on first subplot to avoid clutter
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # Add overall title
    fig.suptitle(f'{model_type.title()} Model', 
                  fontsize=16, y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.savefig(output_dir + f'{model_type}_rows_3d_subplots.png', dpi=300, bbox_inches='tight')    
    print("✓ 3D subplots (2x2 grid) created successfully")


def plot_row_comparison_2d(trace, obs_data, focus_coords, y_pred, model_type):
    """Create 2D comparison plots showing model fit for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']
    
    # Get row effects
    unique_rows = sorted(obs_data['Row'].unique())
    
    # Create figure with contour plots for each row
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Store R² values for summary
    row_r2_values = {}    
    for idx, row in enumerate(unique_rows):
        ax = axes[idx]
        
        # Get row-specific data
        row_data = obs_data[obs_data['Row'] == row]
        row_x = row_data['InterrowDistance'].values
        row_y = row_data['Distance'].values
        row_sev = row_data['Severity'].values
        
        # Get focus coordinates for this row
        focus1_coords = focus_coords[row]['focus1']
        
        # Row-specific intensities
        fx1_row = focus1_coords[0]
        fy1_row = focus1_coords[1]
        fz1_row = summary.loc[f'fz1[{idx}]', 'mean']
       
        # Row-specific prediction function
        def predict_row(X, Y):
            distance = np.sqrt((X - fx1_row)**2 + (Y - fy1_row)**2)
            
            if model_type == 'exponential':
                kernel = fz1_row * np.exp(-distance / scale1_est)

            elif model_type == 'gaussian':
                kernel = fz1_row * np.exp(-(distance**2) / (2 * scale1_est**2))

            elif model_type == 'power_law':
                kernel = fz1_row / np.power(1 + distance / scale1_est, exponent1_est)
            
            return kernel
        
        "calculate R² for this row from posterior predictive"
        row_mask = obs_data['Row'] == row
        y_pred_row = y_pred[row_mask.values]
        r2_row = r2_score(row_sev, y_pred_row)
        row_r2_values[row] = r2_row
        
        # Create grid for contour
        x_min, x_max = row_x.min() - 2, row_x.max() + 2
        y_min, y_max = row_y.min() - 10, row_y.max() + 10
        
        X_grid, Y_grid = np.meshgrid(
            np.linspace(-15, 15, 50),
            np.linspace(y_min-5, y_max+5, 50)
        )
        
        # Create contour plot
        Z_pred = predict_row(X_grid, Y_grid)
        contour = ax.contourf(X_grid, Y_grid, Z_pred, levels=15, cmap='viridis', alpha=0.7)
        
        # Plot observed data
        scatter = ax.scatter(row_x, row_y, c=row_sev, s=80, cmap='viridis', 
                           edgecolors='black', linewidth=0.8, zorder=5)
        
        # Mark foci
        ax.scatter([fx1_row], [fy1_row], 
                  marker='*', s=200, color='red', 
                  edgecolors='darkred', linewidth=2, zorder=6, label='Focus 1')
        
        ax.set_xlabel('InterrowDistance')
        ax.set_ylabel('Distance')
        ax.set_title(f'Row {row}: R² = {r2_row:.3f} (fz1={fz1_row:.3f}')
        ax.legend()
        
        # Add colorbar for this subplot
        plt.colorbar(contour, ax=ax, label='Predicted Severity')
    
    fig.suptitle(f'{model_type.title()} Model', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_dir + f'{model_type}_row_comparison_2d.png', dpi=300, bbox_inches='tight')
    
    # Print R² summary
    print(f"\nRow-specific R² values for {model_type} model:")
    for row, r2 in row_r2_values.items():
        print(f"  Row {row}: R² = {r2:.3f}")
    
    mean_r2 = np.mean(list(row_r2_values.values()))
    print(f"  Mean R² across rows: {mean_r2:.3f}")    
    print("✓ Row comparison 2D plots created successfully")

        

def plot_surfaces(trace, obs_data, focus_coords, y_pred, model_type):
    """Create all surface plots - focused on row-specific analysis"""
    print(f"\nCreating surface plots for {model_type} model...")
    
    # Create 2D row comparison plots (returns R² values)
    print("Creating 2D row comparison plots...")
    plot_row_comparison_2d(trace, obs_data, focus_coords, y_pred, model_type)
    
    # Create comprehensive 3D plot of all rows
    print("Creating 3D plot of all rows...")
    plot_3d_rows_subplots(trace, obs_data, focus_coords, y_pred, model_type)



def model_performance(traces, obs_data, y_pred, focus_coords):
    """
    model selection based on loo, waic
    """    
    "compare with loo"
    loo_comparison = az.compare(traces,ic='loo')
    
    comparisons = {}
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
                
        "Calculate comprehensive model performance metrics for each row and overall data"            
        # Extract parameter estimates
        summary = az.summary(trace, hdi_prob=0.95)
        
        # Get shared parameters
        scale1_est = summary.loc['scale1', 'mean']    
        if model_name == "power_law":
            exponent1_est = summary.loc['exponent1', 'mean']
        
        # Get row effects
        unique_rows = sorted(obs_data['Row'].unique())
        
        # Calculate performance metrics for each row
        performance_metrics = {}
        all_observed = []  # Keep as Python list
        
        for idx, row in enumerate(unique_rows):
            # Get row-specific data
            row_data = obs_data[obs_data['Row'] == row]
            row_x = row_data['InterrowDistance'].values
            row_y = row_data['Distance'].values
            row_sev = row_data['Severity'].values
            
            # Get focus coordinates for this row
            focus1_coords = focus_coords[row]['focus1']
            
            # Row-specific intensities
            fx1_row = focus1_coords[0]
            fy1_row = focus1_coords[1]
            fz1_row = summary.loc[f'fz1[{idx}]', 'mean']
            
            # Calculate predictions and metrics
            row_mask = obs_data['Row'] == row
            y_pred_row = y_pred[row_mask.values]            
            predicted_sev = y_pred_row # posterior predictive
                        
            # Store for overall calculation (extend lists, don't append arrays)
            all_observed.extend(row_sev)
            
            # R²
            r2 = r2_score(row_sev, predicted_sev)
            
            # RMSE
            rmse = np.sqrt(mean_squared_error(row_sev, predicted_sev))
            
            # MAE
            mae = mean_absolute_error(row_sev, predicted_sev)
            
            # Correlation - handle edge cases
            if len(row_sev) > 1 and np.var(row_sev) > 1e-10 and np.var(predicted_sev) > 1e-10:
                correlation = np.corrcoef(row_sev, predicted_sev)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
            
            performance_metrics[row] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Correlation': correlation,
                'n_points': len(row_sev),
                'fz1_intensity': fz1_row
            }
        
        # Calculate overall metrics after processing all rows
        all_observed = np.array(all_observed)
        all_predicted = y_pred
        
        overall_R2 = r2_score(all_observed, all_predicted)
        overall_RMSE = np.sqrt(mean_squared_error(all_observed, all_predicted))
        overall_MAE = mean_absolute_error(all_observed, all_predicted)
        overall_n_points = len(all_observed)
        
        # Calculate overall correlation
        if len(all_observed) > 1 and np.var(all_observed) > 1e-10 and np.var(all_predicted) > 1e-10:
            overall_correlation = np.corrcoef(all_observed, all_predicted)[0, 1]
            if np.isnan(overall_correlation):
                overall_correlation = 0
        else:
            overall_correlation = 0
                
        comparisons[model_name] = {'n_obs':n_obs, 
                               'elpd_loo': elpd_loo, 
                               'p_loo': p_loo, 'rank_loo':rank_loo,
                               'se_d_loo':se_d_loo, 'warning':warning, 
                               'pareto_k_bad':pareto_k_bad, 'overall_R2':overall_R2,
                               'overall_RMSE':overall_RMSE, 'overall_MAE':overall_MAE,
                               'overall_correlation':overall_correlation, 
                               'overall_n_points':overall_n_points,
                               'rows':performance_metrics}
        
    comparisons_df = pd.DataFrame(comparisons)
    
    return comparisons_df
        


def plot_posterior_predictive(y_sim, y_obs, distances, model_type):
    """
    Plot posterior predictive checks for ZOIB model.
    
    Parameters:
    -----------
    y_sim : array (chains, draws, n_obs)
        Posterior predictive samples
    y_obs : array (n_obs,)
        Observed severity values
    distances : array (n_obs,)
        Distance from focus for spatial plot
    model_type : str
        Model name for title
    sim : int
        Simulation ID for filename
    """
    
    # Flatten simulations
    y_sim_flat = y_sim.reshape(-1, len(y_obs))  # (n_samples, n_obs)
    
    # Calculate overall metrics
    y_pred = y_sim_flat.mean(axis=0)
    y_pred_lower = np.percentile(y_sim_flat, 2.5, axis=0)   # Lower 95% CI
    y_pred_upper = np.percentile(y_sim_flat, 97.5, axis=0)  # Upper 95% CI
    
    r2 = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    
    # Create PPC plots
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    
    # 1. Overall distribution
    ax1 = axes[0]
    ax1.hist(y_obs, bins=50, alpha=0.5, label='Observed', density=True, color='red')
    for i in range(min(100, y_sim_flat.shape[0])):
        ax1.hist(y_sim_flat[i], bins=50, alpha=0.01, color='blue', density=True)
    
    ax1.text(-0.05, 1.05, 'A', transform=ax1.transAxes,
                fontsize=16, fontweight='bold', va='top')
    ax1.set_xlabel('Severity', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'Overall Distribution (R² = {r2:.3f})', fontsize=12, fontweight='bold')
    
    legend_handles = [
    mpatches.Patch(color='red', alpha=0.5, label='Observed'),
    mpatches.Patch(color='blue', alpha=1.0, label='Predicted') # Solid color!
    ]
    
    ax1.legend(handles=legend_handles, fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Zero/One inflation
    ax2 = axes[1]
    prop_zeros_obs = np.mean(y_obs == 0)
    prop_ones_obs = np.mean(y_obs == 1)
    prop_zeros_sim = np.mean(y_sim_flat == 0, axis=1)
    prop_ones_sim = np.mean(y_sim_flat == 1, axis=1)
    
    ax2.axvline(prop_zeros_obs, color='red', linewidth=2, 
                label=f'Obs. zeros ({prop_zeros_obs:.1%})', linestyle='--')
    ax2.hist(prop_zeros_sim, bins=30, alpha=0.5, color='red', 
             label=f'Est. zeros (μ={prop_zeros_sim.mean():.1%})')
    ax2.axvline(prop_ones_obs, color='blue', linewidth=2, 
                label=f'Obs. ones ({prop_ones_obs:.1%})', linestyle='--')
    ax2.hist(prop_ones_sim, bins=30, alpha=0.5, color='blue',
             label=f'Est. ones (μ={prop_ones_sim.mean():.1%})')
    
    ax2.text(-0.05, 1.05, 'B', transform=ax2.transAxes,
                fontsize=16, fontweight='bold', va='top')
    ax2.set_xlabel('Proportion', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Zero/One Inflation Check', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Spatial gradient
    ax3 = axes[2]
    
    # Create distance bins (adjust bin width as needed)
    bin_width = 2  # 2-meter bins
    dist_bins = np.arange(0, distances.max() + bin_width, bin_width)
    dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
    
    # Bin observed and predicted
    obs_binned = []
    pred_binned = []
    pred_lower_binned = []
    pred_upper_binned = []
    
    for i in range(len(dist_bins) - 1):
        mask = (distances >= dist_bins[i]) & (distances < dist_bins[i+1])
        if mask.sum() > 0:
            obs_binned.append(y_obs[mask].mean())
            pred_binned.append(y_pred[mask].mean())
            pred_lower_binned.append(y_pred_lower[mask].mean())
            pred_upper_binned.append(y_pred_upper[mask].mean())
    
    # Convert to arrays
    obs_binned = np.array(obs_binned)
    pred_binned = np.array(pred_binned)
    pred_lower_binned = np.array(pred_lower_binned)
    pred_upper_binned = np.array(pred_upper_binned)
    
    # Plot binned data - smooth curves now!
    ax3.scatter(dist_centers, obs_binned, alpha=0.6, label='Observed', 
                s=50, color='black', zorder=3, edgecolors='white', linewidth=0.5)
    ax3.plot(dist_centers, pred_binned, 'b-', linewidth=2, 
             label='Predicted mean', alpha=0.8, zorder=2)
    ax3.fill_between(dist_centers, pred_lower_binned, pred_upper_binned,
                     alpha=0.3, color='blue', label='95% PI', zorder=1)
    
    ax3.text(-0.05, 1.05, 'C', transform=ax3.transAxes,
                fontsize=16, fontweight='bold', va='top')
    ax3.set_xlabel('Distance (y)', fontsize=11)
    ax3.set_ylabel('Severity', fontsize=11)
    ax3.set_title(f'Spatial Gradient (RMSE = {rmse:.3f})', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Continuous values only
    ax4 = axes[3]
    mask_cont = (y_obs > 0) & (y_obs < 1)
    if mask_cont.sum() > 5:
        y_obs_cont = y_obs[mask_cont]
        y_sim_cont = y_sim_flat[:, mask_cont]
        
        ax4.hist(y_obs_cont, bins=30, alpha=0.5, label='Observed', 
                 density=True, color='red')
        for i in range(min(100, y_sim_cont.shape[0])):
            ax4.hist(y_sim_cont[i], bins=30, alpha=0.01, color='blue', density=True)
        
        # R² for continuous only
        y_pred_cont = y_sim_cont.mean(axis=0)
        r2_cont = r2_score(y_obs_cont, y_pred_cont)
                
        ax4.text(-0.05, 1.05, 'D', transform=ax4.transAxes,
                    fontsize=16, fontweight='bold', va='top')
        ax4.set_xlabel('Severity (continuous only)', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title(f'Continuous Values (R² = {r2_cont:.3f})', 
                     fontsize=12, fontweight='bold')
        ax4.legend(handles=legend_handles, fontsize=10)
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filename = f'posterior_predictive_check_{model_type}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')



        

if __name__ == "__main__":

    import multiprocessing
    multiprocessing.freeze_support()
    
    # # HPC environment
    # n_cores = min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    n_cores = min(2, os.cpu_count())

    "datasets"
    output_dir = "./output/"
    
    """Main execution function for multi-row analysis"""        
    print("Loading multi-row disease severity data...")
    
    output_file = output_dir + 'ppc_metrics_all_sims.csv'
    pd.DataFrame(columns=['model_type', 'overall_r2', 'overall_rmse', 'overall_mae',
                          'obs_zeros', 'pred_zeros', 'obs_ones', 'pred_ones', 'continuous_r2', 
                          'continuous_rmse']).to_csv(output_file, index=False) # store all ppc metrics    
    
    df = pd.read_csv('disease_severity_of_year_2023_3rd_survey_onefocus.csv')
    
    "clean data after the first survey"        
    df = quality_control_check(df)

    select_rows = [1, 12, 16]
    df = df[df['Row'].isin(select_rows)]
    
    obs_data, focus_coords = load_and_process_multi_row_data(df)
    print("\nDataset summary:")
    print(f"Total observations: {len(obs_data)}")
    print(f"Rows analyzed: {sorted(obs_data['Row'].unique())}")
    print(obs_data.groupby('Row')['Severity'].describe())
    
    # Test different models
    models_to_test = ['exponential', 'gaussian', 'power_law']
    traces = {}
    
    for model_type in models_to_test:
        print(f"\n{'='*60}")
        print(f"FITTING {model_type.upper()} MODEL (MULTI-ROW)")
        print(f"{'='*60}")
                    
        model = create_multi_row_zoib_model(obs_data, focus_coords, model_type=model_type)            
        with model:
            initial_point = model.initial_point()
            print("✓ Initial point computed successfully")
            
            logp_fn = model.compile_logp()
            initial_logp = logp_fn(initial_point)
            print(f"✓ Initial log-probability: {initial_logp:.4f}")
            
            if not np.isfinite(initial_logp):
                print("⚠ WARNING: Initial log-probability is not finite")
                
            "sample from posterior"
            print("\n--- Starting MCMC sampling ---")                    
            trace = pm.sample(
                draws=8000,
                tune=4000,
                chains=4,
                cores=4,
                target_accept=0.85,
                return_inferencedata=True,
                random_seed=42,
                progressbar=False,
                idata_kwargs={"log_likelihood": True},
                init='advi+adapt_diag'  # Better initialization
                )
            print("✓ Sampling completed!")
            
            "store trace to dict"                        
            traces[model_type] = trace
            
            "save trace file for later plotting"
            trace.to_netcdf(output_dir + f'{model_type}_trace.nc')
                                
            "posterior predictive check"
            # generate posterior predictive samples
            ppc = pm.sample_posterior_predictive(trace, var_names=['obs'], random_seed=42)
            
            # extract generated data
            y_sim = ppc.posterior_predictive['obs'].values                
            y_obs = obs_data['Severity'].values
            distances = obs_data['Distance'].values
            
            # plot posterior predictive check
            plot_posterior_predictive(y_sim, y_obs, distances, model_type)                
            y_pred = ppc.posterior_predictive['obs'].mean(dim=['chain', 'draw']).values
            y_low = ppc.posterior_predictive['obs'].quantile(0.025, dim=['chain', 'draw']).values
            y_high = ppc.posterior_predictive['obs'].quantile(0.975, dim=['chain', 'draw']).values
            
            "calculate ppc metrics"
            mask_cont = (y_obs > 0) & (y_obs < 1)
            y_obs_cont = y_obs[mask_cont]
            y_pred_cont = y_pred[mask_cont]
            
            metrics = {
                    'model_type': model_type,
                    'overall_r2': r2_score(y_obs, y_pred),
                    'overall_rmse': np.sqrt(mean_squared_error(y_obs, y_pred)),
                    'overall_mae': mean_absolute_error(y_obs, y_pred),
                    
                    # Zero/One inflation
                    'obs_zeros': np.mean(y_obs == 0),
                    'pred_zeros': np.mean(y_sim == 0),
                    'obs_ones': np.mean(y_obs == 1),
                    'pred_ones': np.mean(y_sim == 1),
                    
                    # Continuous values only
                    'continuous_r2': r2_score(y_obs_cont, y_pred_cont) if mask_cont.sum() > 5 else np.nan,
                    'continuous_rmse': np.sqrt(mean_squared_error(y_obs_cont, y_pred_cont)) if mask_cont.sum() > 5 else np.nan,
                }
                
            pd.DataFrame([metrics]).to_csv(output_file, mode='a', header=False, index=False)
                                
            
            "check convergence"
            diagnostics, summary = convergence_diagnostics(trace)
            print(diagnostics)
            diagnostics.to_csv(output_dir + f'{model_type}_disgnostic.csv')
            summary.to_csv(output_dir + f'{model_type}_trace_summary.csv')
                                
            "create plots if diagnostics worked"                    
            print("\nCreating result plots...")
            plot_multi_row_results(trace, obs_data, focus_coords, y_pred, y_low, y_high, model_type)
            print("✓ Plots created successfully")
            
            # Add surface plots with row-specific R²
            plot_surfaces(trace, obs_data, focus_coords, y_pred, model_type)
            print("✓ Surface plots created successfully")
                    
            "calculate effective sample sizes"
            ess_df = calculate_all_ess(trace)
            ess_df['ess_num'] = ess_df['ess'].astype(str).str.extract(r'(\d+\.\d+e[+-]\d+)>')
            ess_df['ess_num'] = pd.to_numeric(ess_df['ess_num'])
            ess_df.to_csv(output_dir + f"{model_type}_effective_sample_size.csv")                        
                                        
    
    "compare models if multiple successful"
    if len(traces) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        loo_comparison = az.compare(traces, ic='loo')
        comp_df = model_performance(traces, obs_data, y_pred, focus_coords)
        comp_df.to_csv(output_dir + "models_performances.csv")
        
        print("\nLOO comparison:")
        print(loo_comparison)
        # Get the best model
        best_model = loo_comparison.index[0]  # First row is best model
        best_loo = loo_comparison.loc[best_model, 'elpd_loo']            
        print(f"\n Best model: {best_model}")
        print(f"   LOO score: {best_loo:.1f}")
        
        # Show differences from best model
        print("\nModel ranking by predictive performance:")
        for i, (model, row) in enumerate(loo_comparison.iterrows()):
            rank = i + 1
            loo_val = row['elpd_loo']
            elpd_diff = row['elpd_diff']
            se = row['dse']
            
            if rank == 1:
                print(f"  {rank}. {model}: LOO = {loo_val:.1f} (Best model)")
            else:
                print(f"  {rank}. {model}: LOO = {loo_val:.1f} (Δ = {elpd_diff:.1f} ± {se:.1f})")
                
                # Interpret the difference
                if elpd_diff > 2 * se:
                    strength = "Strong evidence against"
                elif elpd_diff > se:
                    strength = "Moderate evidence against"
                else:
                    strength = "Weak evidence against"
                print(f"      → {strength} this model")
        
        "additional LOO diagnostics"
        print("\nLOO Diagnostics:")
        for model_name, trace in traces.items():
            loo_result = az.loo(trace, pointwise=True)
            n_high_pareto = np.sum(loo_result.pareto_k > 0.7)
            n_total = len(loo_result.pareto_k)
            
            print(f"  {model_name}: {n_high_pareto}/{n_total} high Pareto-k values")
            if n_high_pareto > 0:
                print(f"    → {n_high_pareto/n_total*100:.1f}% potentially problematic points")

            # show which models completed successfully
            print(f"\nSuccessfully fitted models: {list(traces.keys())}")
            
        "try WAIC if LOO fails"
        print(f"\n{'='*60}")
        print("WAIC COMPARISON (Alternative Information Criterion)")
        print(f"{'='*60}")
        
        waic_comparison = az.compare(traces, ic='waic')
        print("\nWAIC Comparison (lower = better):")
        print(waic_comparison)

