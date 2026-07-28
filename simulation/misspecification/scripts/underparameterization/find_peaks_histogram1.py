# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 16:37:59 2026

@author: sunny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import arviz as az


def find_modes_histogram(samples, n_bins=60):
    """
    Find modes directly from histogram (no smoothing)
    Most accurate for sharp, well-separated peaks
    """
    counts, bin_edges = np.histogram(samples, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks in histogram
    # Lower prominence threshold to catch smaller peaks
    peaks, properties = find_peaks(counts, prominence=0.05*counts.max())
    
    if len(peaks) == 0:
        return [], []
    
    peak_locations = bin_centers[peaks]
    peak_heights = counts[peaks]
    
    # Sort by height (tallest first)
    sorted_idx = np.argsort(peak_heights)[::-1]
    
    return peak_locations[sorted_idx], peak_heights[sorted_idx]



start_sim = 50 
end_sim = 100
output_dir = "./continue1/output/"
separation_threshold = 30
all_peaks_df = pd.DataFrame()
for sim in range(start_sim, end_sim):
    trace_path = f"{output_dir}power_law_trace_{sim}.nc"    
    trace = az.from_netcdf(trace_path)
    
    # Extract fy1 samples
    fy1_data = trace.posterior['fy1']
    n_rows = fy1_data.shape[-1]
    
    "plot histogram and peaks"
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4*n_rows))

    results = []    
    for row_idx in range(n_rows):
        ax = axes[row_idx]
        
        # Extract samples for this row
        samples = fy1_data.sel(fy1_dim_0=row_idx).values.flatten()
        
        # Plot histogram
        counts, bins, patches = ax.hist(samples, bins=60, density=True, 
                                         alpha=0.7, color='steelblue', 
                                         edgecolor='black', linewidth=0.5)
        
        # Find histogram peaks
        hist_modes, hist_heights = find_modes_histogram(samples, n_bins=60)
        
        # Mark peaks with different colors
        colors = ['red', 'orange', 'purple', 'brown']
        for i, mode in enumerate(hist_modes[:4]):  # Show up to 4 peaks
            color = colors[i] if i < len(colors) else 'gray'
            ax.axvline(mode, color=color, linestyle='--', linewidth=2.5,
                      label=f'Peak {i+1}: Y={mode:.1f}', alpha=0.9)
        
        # Add statistics
        mean_val = samples.mean()
        std_val = samples.std()
        median_val = np.median(samples)
        
        ax.axvline(mean_val, color='blue', linestyle=':', linewidth=2,
                  alpha=0.6, label=f'Mean: {mean_val:.1f}')
        
        # Labels and title
        ax.set_xlabel('Focus Location fy1[row] (Y coordinate)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        
        # Determine modality
        if len(hist_modes) == 1:
            modality_str = "UNIMODAL"
        elif len(hist_modes) == 2:
            modality_str = "BIMODAL"
        elif len(hist_modes) >= 3:
            modality_str = f"MULTIMODAL ({len(hist_modes)} peaks)"
        else:
            modality_str = "NO CLEAR PEAKS"
        
        ax.set_title(f'Row {row_idx}: fy1 Posterior Distribution ({modality_str})\n'
                    f'Mean={mean_val:.1f}, Median={median_val:.1f}, SD={std_val:.1f}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Assess bimodality
        is_bimodal = len(hist_modes) >=2
        
        if is_bimodal:
            # Calculate separation between two largest peaks
            peak1, peak2 = hist_modes[0], hist_modes[1]
            separation = abs(peak1 - peak2)
            is_well_separated = separation > separation_threshold # Here the threshold is 30
        else:
            separation = 0
            is_well_separated = False
        
        # Store results
        results.append({
            'simulation':sim,
            'row': row_idx,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'n_peaks': len(hist_modes),
            'peak_locations': hist_modes[:4].tolist() if len(hist_modes) > 0 else [],
            'sorted_peak_locations':sorted(hist_modes),
            'is_bimodal': is_bimodal,
            'is_well_separated': is_well_separated,
            'separation': separation
        })
        

    plt.tight_layout()
    output_path = trace_path.replace('.nc', '_fy1_histogram_peaks.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    
    peaks_df = pd.DataFrame(results)
    all_peaks_df = pd.concat([all_peaks_df, peaks_df])
all_peaks_df.to_csv(f"power_law_peaks_in_posterior_distribution_{start_sim}_{end_sim}.csv")