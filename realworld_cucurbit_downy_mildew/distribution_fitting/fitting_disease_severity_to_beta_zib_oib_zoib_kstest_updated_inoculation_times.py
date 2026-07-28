# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:23:02 2025

@author: sunny
"""

import pandas as pd
import numpy as np
from scipy.stats import beta, ecdf, bootstrap
from scipy.optimize import minimize, differential_evolution
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import seaborn as sns


"""
Liu & Kong (2015)
f(y) = { p                    if y = 0     (point mass)
       { (1-p)·q                    if y = 1     (point mass)  
       { (1-p)·(1-q)·Beta(y;α,β) if y ∈ (0,1) (continuous density)
"""

def zero_one_inflated_logpdf(x, p, q, alpha_para, beta_para):
    x = np.asarray(x)        
    logpdf = np.full_like(x, -np.inf, dtype=float)
    
    "point mass at 0"
    zero_mask = (x == 0)
    if np.any(zero_mask):
        logpdf[zero_mask] = np.log(p)
    
    "point mass at 1"  
    one_mask = (x == 1)
    if np.any(one_mask):
        logpdf[one_mask] = np.log(1-p) + np.log(q)
    
    "continuous Beta component for (0,1)"
    continuous_mask = (x > 0) & (x < 1)
    if np.any(continuous_mask):
        beta_logpdf = beta.logpdf(x[continuous_mask], alpha_para, beta_para)
        if np.any(~np.isfinite(beta_logpdf)):
            logpdf[continuous_mask] = -1e10
        else:
            logpdf[continuous_mask] = np.log(1-p) + np.log(1-q) + beta_logpdf
                                        
    return logpdf


def zero_one_inflated_pdf(x, p, q, alpha_para, beta_para):
    """Segmented function for zero-one inflated Beta distribution"""
    x = np.asarray(x)
    prob = np.zeros_like(x, dtype=float)
    
    "point mass at 0"
    zero_mask = (x == 0)
    if np.any(zero_mask):
        prob[zero_mask] = p
    
    "point mass at 1"
    one_mask = (x == 1)
    if np.any(one_mask):
        prob[one_mask] = (1 - p) * q
    
    "continuous Beta component for (0,1)"
    continuous_mask = (x > 0) & (x < 1)
    if np.any(continuous_mask):
        try:
            beta_pdf_vals = beta.pdf(x[continuous_mask], alpha_para, beta_para)
            prob[continuous_mask] = (1 - p) * (1 - q) * beta_pdf_vals
        except:
            prob[continuous_mask] = 1e-12
                
    return prob


def calc_proportion(data):
    data = np.array(data)
    
    "count zeros and ones"
    n0 = np.sum(data == 0)
    n1 = np.sum(data == 1)
    n_total = len(data)
    
    "calculate proportions"
    p0 = n0 / n_total
    p1 = n1 / (n_total - n0) if n_total > n0 else 0  # P(1 | not 0)
    
    "extract data in open interval (0,1)"
    continuous_data = data[(data > 0) & (data < 1)]
    
    return n0, n1, p0, p1, continuous_data



def neg_log_likelihood(params_eta, data):
    # unpack parameters
    eta_p0, eta_p1, eta_mu, log_phi = params_eta
    
    "apply link fnction to get parameters in the [0, 1] range"
    #logit link: P = expit(eta)
    p0_est = expit(eta_p0)
    p1_est = expit(eta_p1) # P(1| not 0)
    mu_est = expit(eta_mu) # mean of the Beta component
    
    #logit link:phi = exp(log_phi)
    phi_est = np.exp(log_phi)
    
    "caculate Beta parameters (Alpha and Beta)"
    a_est = mu_est * phi_est
    b_est = (1 - mu_est) * phi_est
    
    "check phi bounds"
    if phi_est <= 1e-08:
        return 1e10

    "add constraints for extreme conditions"                        
    try:        
        loglikelihood = np.sum(zero_one_inflated_logpdf(data, p0_est, p1_est, a_est, b_est))
        
        if not np.isfinite(loglikelihood):
            return 1e10
            
    except Exception as e:
        print("neg likelihood error", e)
        return 1e10
        
    neglogll = -loglikelihood
    
    return neglogll



def fit_beta(data):
    "initialize parameters"
    n0, n1, p0_init, p1_init, continuous_data = calc_proportion(data)
    
    epsilon = 1e-10
    p0_init = np.clip(p0_init, epsilon, 1 - epsilon)
    p1_init = np.clip(p1_init, epsilon, 1 - epsilon)
    
    if len(continuous_data) == 0:
        mu_init = 0.5
    else:
        mu_init = continuous_data.mean()
        
    mu_init = np.clip(mu_init, epsilon, 1 - epsilon)
    
    "convert initial parameters to logit-space using ligit(p)"
    initial_params_eta = [
        logit(p0_init),
        logit(p1_init),
        logit(continuous_data.mean()), # initial estimate for mu
        np.log(1.0) # initial estimate for log(phi), use log(1)=0 as a neutral start for precision
        ]
    
    "define bounds for parameters (use -inf to inf for logit-space"
    bounds_eta = [
        (-15, 15),  #eta_p0
        (-15, 15),  #eta_p1
        (-15, 15), #eta_mu
        (-5, 10)] #log(phi) (phi itself is > 0)
    
    'minimize negative likelihood'
    result = minimize(neg_log_likelihood, initial_params_eta, args=(data,), 
                      method='L-BFGS-B', bounds=bounds_eta)
    
    if not result.success:
        print("Warning: Optimization did not converge!")
        
    return result



def fit_beta_global(data):
    "initialize parameters"
    n0, n1, p0_init, p1_init, continuous_data = calc_proportion(data)
    
    epsilon = 1e-10
    p0_init = np.clip(p0_init, epsilon, 1 - epsilon)
    p1_init = np.clip(p1_init, epsilon, 1 - epsilon)
    
    if len(continuous_data) == 0:
        mu_init = 0.5
    else:
        mu_init = continuous_data.mean()
        
    mu_init = np.clip(mu_init, epsilon, 1 - epsilon)
            
    "convert initial parameters to logit-space using ligit(p)"
    initial_params_eta = [
        logit(p0_init),
        logit(p1_init),
        logit(mu_init), # initial estimate for mu
        np.log(1.0) # initial estimate for log(phi), use log(1)=0 as a neutral start for precision
        ]
    
    "define bounds for parameters (use -inf to inf for logit-space"
    bounds_eta = [
        (-15, 15),  #eta_p0
        (-15, 15),  #eta_p1
        (-15, 15), #eta_mu
        (-5, 10)] #log(phi) (phi itself is > 0)
    
    "global optimization"
    result = differential_evolution(neg_log_likelihood, bounds_eta, args=(data,),
                                    maxiter=100000, popsize=2000, tol=1e-08, seed=42)
    
    return result



def zoib_cdf_theoretical(x, p0_est, p1_est, a_est, b_est):
    """
    Cumulative distribution function for Zero-One Inflated Beta distribution
    """
    x = np.asarray(x)
    x = np.sort(x)
    result = np.zeros_like(x, dtype=float)
    
    zero_mask = (x <= 1e-12)
    one_mask = (x >= 1 - 1e-12)
    continuous_mask = (~zero_mask) & (~one_mask)
    
    "x = 0 (point mass)"
    result[zero_mask] = p0_est
    
    "0 < x < 1 (continuous Beta component)"
    if np.any(continuous_mask):
        pi_beta = (1 - p0_est) * (1 - p1_est)
        result[continuous_mask] = p0_est + pi_beta * beta.cdf(x[continuous_mask], a_est, b_est)
    
    "x >= 1 (point mass)"
    result[one_mask] = 1.0  # should equal to 1
        
    return result



def custom_ks_test_zoib(data, p0_est, p1_est, a_est, b_est, n_bootstrap=10000, random_state=21):
    """
    Custom KS test for Zero-One Inflated Beta distribution
    """
    data = np.asarray(data)
    n = len(data)
    
    def generate_zoib_sample(n_samples, rng=None):
        "generate zoib samples"
        if rng is None:
            rng = np.random.default_rng()
            
        samples = []
        for _ in range(n_samples):
            u = rng.random()
            if u < p0_est:
                samples.append(0.0)
            elif u < p0_est + (1 - p0_est) * p1_est:
                samples.append(1.0)
            else:
                samples.append(rng.beta(a_est, b_est))
        return np.array(samples)

    "calculate ks stat of real data"
    def ks_statistics(sample):
        sample_sorted = np.sort(sample)
        n_sample = len(sample_sorted)
        
        empirical_cdf = np.arange(1, n_sample+1) / n_sample
        theoretical_cdf = zoib_cdf_theoretical(sample_sorted , p0_est, p1_est, a_est, b_est)
        ks_score = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        return ks_score
        
    "bootstrap simulation"
    def bootstrap_statistics(random_state):
        rng = np.random.default_rng(random_state)
        bootstrap_sample = generate_zoib_sample(n, rng)
        
        return ks_statistics(bootstrap_sample)
    
    
    def statistic_wrapper(*args):
        return bootstrap_statistics(None)
    
    
    bootstrap_res = bootstrap(
            (data,), 
            statistic_wrapper,
            n_resamples=n_bootstrap,
            random_state=random_state,
            method='basic'  # Basic bootstrap method
        )
        
    "calculate p-value"
    bootstrap_stats = bootstrap_res.bootstrap_distribution
    
    observed_ks = ks_statistics(data)
    p_value = np.mean(np.array(bootstrap_stats) >= observed_ks)
            
    return observed_ks, p_value, bootstrap_res 




year = '2023'
survey_times = ["3rd"]

"2023"
treatments_2023 = ["CK", "1-Windward-focus-once", "1-Windward-focus-twice", "2V-foci-once", "2V-foci-twice"]
row_names_2023 = [i for i in range(1, 21)]
row_treatments_2023 = ["1-Windward-focus-once", "CK", "2V-foci-twice", "CK", "1-Windward-focus-twice", 
              "1-Windward-focus-twice", "2V-foci-once", "CK", "2V-foci-once", "2V-foci-twice",
              "CK", "1-Windward-focus-once", "2V-foci-twice", "2V-foci-once", "1-Windward-focus-twice",
              "1-Windward-focus-once", "2V-foci-once", "1-Windward-focus-twice", "CK", "2V-foci-twice"]

"combine data for each treatment and calculate the average value"
meta_df_2023 = pd.DataFrame(data={"Year":2023, "Row": row_names_2023, "Treatment":row_treatments_2023})


survey_years = []
surveys = []
treats = []
minimize_fits = []
mini_as = []
mini_bs = []
mini_kss = []
mini_pvals = []
mini_aics = []
mini_bics = []
global_fits = []
glo_as = []
glo_bs = []
glo_mus = []
glo_phis = []
glo_kss = []
glo_pvals = []
glo_aics = []
glo_bics = []
mini_p0s = []
mini_p1s = []
glo_p0s = []
glo_p1s = []
real_p0s = []
real_p1s = []

sns.set_theme(style='darkgrid')
sns.set(font_scale=1.5)
fig, axs = plt.subplots(5, 2, figsize=(18, 20), dpi=600)
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


for survey in survey_times:
    dataset = pd.read_csv('disease_severity_of_year_' + year + '_' + survey + '_survey.csv')    
    treatments = treatments_2023
    meta_df= meta_df_2023
    
    "processing by treatment"
    for i, treatment in enumerate(treatments):
        treatment_rows = meta_df[meta_df["Treatment"]==treatment]["Row"].tolist()
        dataset_treatment = dataset[dataset['Row'].isin(treatment_rows)]
        
        "filter out nan value which cannot be used to fitting"
        dataset_treatment['Severity'] = dataset_treatment['Severity_Count'].astype('float64') / 100.0
        valid_severity = dataset_treatment['Severity'].dropna()
        valid_severity = valid_severity.tolist()
        
        n0, n1, p0, p1, continuous_data = calc_proportion(valid_severity)
                
        "plot original severity histogram"
        sns.set_style('white')
        sns.set_context('paper', font_scale=2)
        sns.displot(data=dataset_treatment, x='Severity', kind='hist', bins=100, aspect=1.5)
        
        res_minimize = fit_beta(valid_severity)
        res_global = fit_beta_global(valid_severity)
        
        
        "fitting performance"
        if res_minimize.success:
            eta_p0_mini, eta_p1_mini, eta_mu_mini, log_phi_mini = res_minimize.x
            p0_mini = expit(eta_p0_mini)
            p1_mini = expit(eta_p1_mini)
            mu_mini = expit(eta_mu_mini)
            phi_mini = np.exp(log_phi_mini)
            
            alpha_mini = mu_mini * phi_mini
            beta_mini = (1 - mu_mini) * phi_mini                
            print(f"fitted parameters of local minimization: a={alpha_mini:.4f}, b={beta_mini:.4f}")    

        else:
            print("Warning: Local minimize optimization did not converge!")
            print("Message:", res_minimize.message)
            p0_mini, p1_mini, alpha_mini, beta_mini = np.nan, np.nan, np.nan, np.nan
                        
        if res_global.success:
            eta_p0_glo, eta_p1_glo, eta_mu_glo, log_phi_glo = res_global.x
            p0_glo = expit(eta_p0_glo)
            p1_glo = expit(eta_p1_glo)
            mu_glo = expit(eta_mu_glo)
            phi_glo = np.exp(log_phi_glo)
            
            alpha_glo = mu_glo * phi_glo
            beta_glo = (1 - mu_glo) * phi_glo                
            print(f"fitted parameters of local minimization: a={alpha_glo:.4f}, b={beta_glo:.4f}")    
            
        else:
            print("Warning: Global minimize optimization did not converge!")
            print("Message:", res_global.message)
            p0_glo, p1_glo, alpha_glo, beta_glo = np.nan, np.nan, np.nan, np.nan
        
               
        "plot the fitted function with the real data frequency"        
        if res_minimize.success:
            ksstat_mini, pval_mini, btres_mini = custom_ks_test_zoib(valid_severity, p0_mini, p1_mini, alpha_mini, beta_mini)

                
        if res_global.success:
            ksstat_glo, pval_glo, btres_glo = custom_ks_test_zoib(valid_severity, p0_glo, p1_glo, alpha_glo, beta_glo)
            
                           
            "plot density and point mass"
            "continuous part"
            if len(continuous_data) > 0:
                x_cont = np.linspace(0.001, 0.999, 1000)
                prob_cont = zero_one_inflated_pdf(x_cont, p0_glo, p1_glo, alpha_glo, beta_glo)
                axs[i][0].plot(x_cont, prob_cont, 'k-', linewidth=2, label='Estimated Continuous PDF')
                
                axs[i][0].hist(continuous_data, bins=30, density=True, alpha=0.3, color='blue',
                        edgecolor='black', label='Observed Continuous Frequency')
            
            "point masses as bars"
            point_probs = [p0, (1-p0)*p1]
            theoretical_probs = [zero_one_inflated_pdf(0, p0_glo, p1_glo, alpha_glo, beta_glo),
                                zero_one_inflated_pdf(1, p0_glo, p1_glo, alpha_glo, beta_glo)]
            
            ylim_min = max(max(point_probs), max(theoretical_probs)) + 0.01
            
            ax_point = axs[i][0].twinx()
            ax_point.plot([0, 1], point_probs, 'bo', label='Observed Point Mass')

            ax_point.plot([0.02, 0.98], theoretical_probs, 'kD', label='Estimated Point Mass')
            ax_point.set_ylim(top=ylim_min)
                           
            ax_point.set_ylabel('Probability')
            axs[i][0].set_xlabel('Severity')
            axs[i][0].set_ylabel('Density')
            
            lines1, labels1 = axs[i][0].get_legend_handles_labels()
            lines2, labels2 = ax_point.get_legend_handles_labels()
            axs[i][0].legend(lines1 + lines2, labels1 + labels2) 
            axs[i][0].grid(True, alpha=0.3)
            axs[i][0].text(-0.1, 1.05, alphabets[i], transform=axs[i][0].transAxes, 
                       fontsize=20, fontweight='bold', va='bottom', ha='right')
        
            
            "plot goodness of fit"
            sorted_valid_severity = np.sort(valid_severity)
            ecdf_res = ecdf(valid_severity)
            ecdf_res.cdf.plot(axs[i][1], label='Empirical CDF')
            theoretical_cdf_glo = zoib_cdf_theoretical(sorted_valid_severity, p0_glo, p1_glo, alpha_glo, beta_glo)
            axs[i][1].plot(sorted_valid_severity, theoretical_cdf_glo, 'k-', linewidth=2, markersize=6, 
                    label='Estimated CDF')
            
            axs[i][1].set_xlabel('Severity')
            axs[i][1].set_ylabel('Probability')
            axs[i][1].legend()
            axs[i][1].grid(True, alpha=0.3)
            axs[i][1].text(0.6, 0.5, f'KS statistic: {ksstat_glo:.3f}, \nP = {pval_glo:.3f}', transform=axs[i][1].transAxes, 
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))

            axs[i][1].text(-0.1, 1.05, alphabets[i+5], transform=axs[i][1].transAxes, 
                       fontsize=20, fontweight='bold', va='bottom', ha='right')


        if res_global.success:
            logll_glo = -neg_log_likelihood([eta_p0_glo, eta_p1_glo, eta_mu_glo, log_phi_glo], valid_severity)
            aic_glo = -2 * logll_glo + 2 * 4 # 4 parameters
            bic_glo = -2 * logll_glo + 4 * np.log(len(valid_severity))
            
            print(f"AIC:{aic_glo:.2f}, BIC: {bic_glo:.2f}")
        else:      
            ksstat_glo = np.nan
            pval_glo = np.nan
            logll_glo = np.nan
            aic_glo = np.nan
            bic_glo = np.nan
            
                    
        survey_years.append(year)
        surveys.append(survey)
        treats.append(treatment)
        global_fits.append(res_global.success)
        glo_as.append(alpha_glo)
        glo_bs.append(beta_glo)
        glo_mus.append(mu_glo)
        glo_phis.append(phi_glo)
        glo_p0s.append(p0_glo)
        glo_p1s.append(p1_glo)
        glo_kss.append(ksstat_glo)
        glo_pvals.append(pval_glo)
        glo_aics.append(aic_glo)
        glo_bics.append(bic_glo)
        real_p0s.append(p0)
        real_p1s.append(p1)


"save figure"
fig.tight_layout()
fig.subplots_adjust(hspace=0.25)
fig.savefig("beta_fitting_" + year + "_" + survey + "_global1.png", dpi=600)


"save data"
fitting_data = pd.DataFrame(data={'Year':survey_years, 'Survey':surveys, 'Treatment':treats, 'real_p0':real_p0s, 'real_p1':real_p1s, 'Global':global_fits, 'glo_Alpha':glo_as,
                                  'glo_Beta':glo_bs, 'glo_mu':glo_mus, 'glo_phi':glo_phis, 'glo_p0':glo_p0s, 'glo_p1':glo_p1s, 'glo_KS':glo_kss, 'glo_p value':glo_pvals,
                                  'glo_AIC':glo_aics, 'glo_BIC':glo_bics})

fitting_data.to_csv(year + "_Surve_treatment_severity_fitting_beta_distribution1.csv")        
            
    
    
