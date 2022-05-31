import copy
import time
import pystan
import numpy as np
import scipy as sp
from pystan.misc import _array_to_table


def linear_control_variates(samples, grad_log_prob):
    try:
        dim = samples.shape[1]
        control = -0.5 * grad_log_prob

        sc_matrix = np.concatenate((samples, control), axis=1)
        sc_cov = np.cov(sc_matrix.T)
        Sigma_cs = sc_cov[0:dim, dim:dim*2].T
        Sigma_cc = sc_cov[dim:dim*2, dim:dim*2]

        inv_Sigma_cc = sp.linalg.inv(Sigma_cc)
        zv = (-inv_Sigma_cc @ Sigma_cs).T @ control.T
        linear_cv_samples = samples + zv.T
    except:
        linear_cv_samples = np.empty_like(samples)
        linear_cv_samples[:] = np.nan
    return linear_cv_samples


def quadratic_control_variates(constrained_samples, unconstrained_samples, grad_log_prob):
    try:
        num_samples_total = constrained_samples.shape[0]
        dim_constrained_samples = constrained_samples.shape[1]
        dim_unconstrained_samples = unconstrained_samples.shape[1]
        
        if dim_unconstrained_samples < 50:
            dim_cp = int(0.5*dim_unconstrained_samples*(dim_unconstrained_samples-1))
            dim_control = dim_unconstrained_samples+dim_unconstrained_samples+dim_cp
            z = -0.5 * grad_log_prob
            control = np.concatenate((z, (unconstrained_samples*z - 0.5)), axis=1)
            control_parts = np.zeros((num_samples_total, dim_cp))
            for i in range(2, dim_unconstrained_samples+1):
                for j in range(1, i):
                    ind = int(0.5*(2*dim_unconstrained_samples-j)*(j-1)) + (i-j)
                    control_parts[:,ind-1] = unconstrained_samples[:,i-1]*z[:,j-1] + unconstrained_samples[:,j-1]*z[:,i-1]
            control = np.concatenate((control, control_parts), axis=1)
        else:
            print('WARNING... The dimentionality of the problem is too large ( > 50), using reduced control variates for the quadratic version.')
            dim_control = dim_unconstrained_samples+dim_unconstrained_samples
            z = -0.5 * grad_log_prob
            control = np.concatenate((z, (unconstrained_samples*z - 0.5)), axis=1)
        
        sc_matrix = np.concatenate((constrained_samples.T, control.T), axis=0)
        sc_cov = np.cov(sc_matrix)
        Sigma_cs = sc_cov[0:dim_constrained_samples, dim_constrained_samples:dim_constrained_samples+dim_control].T
        Sigma_cc = sc_cov[dim_constrained_samples:dim_constrained_samples+dim_control, dim_constrained_samples:dim_constrained_samples+dim_control]

        inv_Sigma_cc = sp.linalg.inv(Sigma_cc)
        zv = (-inv_Sigma_cc @ Sigma_cs).T @ control.T
        quad_cv_samples = constrained_samples + zv.T
    except:
        quad_cv_samples = np.empty_like(constrained_samples)
        quad_cv_samples[:] = np.nan

    return quad_cv_samples


def run_postprocess(fit, cv_mode='linear', permuted=False):
    num_chains = fit.sim['chains']
    num_save = fit.sim['n_save']
    num_warmups = fit.sim['warmup2']
    num_samples_total = 0
    for ns, nw in zip(num_save, num_warmups):
        num_samples_total += ns - nw

    ## Collect unconstrained parameters for grad_log_prob()
    parameter_names = copy.copy(fit.sim['pars_oi'])
    parameter_names.remove('lp__')    # not a parameter in the stan model
    parameter_extract = fit.extract(parameter_names, permuted=permuted)

    # Different data structures when permuted=True or permuted=False
    if permuted == False:
        for parameter_name in parameter_names:
            paramter_tmp = []
            for ci in range(num_chains):
                paramter_tmp.append(parameter_extract[parameter_name][:, ci])
            parameter_extract[parameter_name] = np.concatenate(paramter_tmp, axis=0)

    # Unconstraint mcmc samples.
    unconstrained_mcmc_samples = []
    constrained_mcmc_samples = []
    for i in range(num_samples_total):
        tmp_dict = {}
        tmp_constrained = []
        for parameter_name in parameter_names:
            tmp_extract = parameter_extract[parameter_name][i]
            if isinstance(tmp_extract, np.ndarray):
                tmp_dict[parameter_name] = np.squeeze(tmp_extract)
                if tmp_extract.ndim > 1:
                    tmp_constrained.extend(list(tmp_dict[parameter_name].flatten('F')))
                else:
                    tmp_constrained.extend(list(tmp_dict[parameter_name]))
            else:
                tmp_dict[parameter_name] = tmp_extract
                tmp_constrained.extend([tmp_dict[parameter_name]])
        constrained_mcmc_samples.append(tmp_constrained)
        unconstrained_mcmc_samples.append(fit.unconstrain_pars(tmp_dict))
    unconstrained_mcmc_samples = np.array(unconstrained_mcmc_samples)
    constrained_mcmc_samples = np.array(constrained_mcmc_samples)

    # constrained parameter indices
    constrained_dim = []
    param_dims = unconstrained_mcmc_samples.shape[1]
    for d in range(param_dims):
        check = unconstrained_mcmc_samples[:, d] != constrained_mcmc_samples[:, d]
        if np.any(check):
            constrained_dim.append(d)
    constrained_dim = np.array(constrained_dim)

    # Calculate gradients of the log-probability
    grad_start_time = time.time()
    grad_log_prob_vals = []
    for i in range(num_samples_total):
        grad_log_prob_vals.append(fit.grad_log_prob(unconstrained_mcmc_samples[i], adjust_transform=True))
    grad_log_prob_vals = np.array(grad_log_prob_vals)
    grad_runtime = time.time() - grad_start_time

    cv_start_time = time.time()
    if cv_mode == 'linear':
        # Run control variates
        cv_samples = linear_control_variates(constrained_mcmc_samples, grad_log_prob_vals)
        cv_runtime = time.time() - cv_start_time
        # print('Gradient time: {:.05f} --- Linear control variate time: {:.05f}.'.format(grad_runtime, cv_runtime))
    elif cv_mode == 'quadratic':
        cv_samples = quadratic_control_variates(constrained_mcmc_samples, unconstrained_mcmc_samples, grad_log_prob_vals)
        cv_runtime = time.time() - cv_start_time
        # print('Gradient time: {:.05f} --- Quadratic control variate time: {:.05f}.'.format(grad_runtime, cv_runtime))
    else:
        print('The mode of control variates must be linear or quadratic.')
        return None
    
    return cv_samples, (grad_runtime, cv_runtime), constrained_dim
    

"""
Stan summary:
mean: sample means
se_mean: standard error for the mean = sd / sqrt(n_eff)
sd: sample standard deviations
quantiles:
n_eff: effective sample size
Rhat:
"""

def get_cv_sample_mean_std(fit, cv_samples):
    sample_mean = {}
    sample_std = {}
    means = np.mean(cv_samples, axis=0)
    stds = np.std(cv_samples, axis=0)
    fnames = copy.copy(fit.sim['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    assert(len(fnames) == means.shape[0])
    assert(len(fnames) == stds.shape[0])
    for i, fname in enumerate(fnames):
        sample_mean[fname] = means[i]
        sample_std[fname] = stds[i]
    return sample_mean, sample_std

def get_parameter_std(fit, cv_samples, cv_samples_squared):
    parameter_std = {}
    expect_x = np.mean(cv_samples, axis=0)
    expect_x_squared = np.mean(cv_samples_squared, axis=0)
    fnames = copy.copy(fit.sim['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    assert(len(fnames) == expect_x.shape[0])
    assert(len(fnames) == expect_x_squared.shape[0])
    for i, fname in enumerate(fnames):
        parameter_std[fname] = np.sqrt(expect_x_squared[i] - expect_x[i]**2)
    return parameter_std

def get_cv_sample_ess(fit, cv_samples):
    sim_copy = copy.deepcopy(fit.sim)
    num_save = sim_copy['n_save']
    num_warmups = sim_copy['warmup2']
    num_samples = []
    num_samples_total = 0
    for ns, nw in zip(num_save, num_warmups):
        num_samples.append(ns - nw)
        num_samples_total += ns - nw
    fnames = copy.copy(sim_copy['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    assert(len(fnames) == cv_samples.shape[1])
    # reuse Pystan ess function to calculate ess of control variates samples
    # substitute the MCMC samples in fit by control variate samples
    for ci, chain in enumerate(sim_copy['samples']):
        for fi, fname in enumerate(fnames):
            chain['chains'][fname][num_warmups[ci]:num_save[ci]] = cv_samples[ci*num_samples[ci]:(ci+1)*num_samples[ci], fi]
    ess = {}
    for fi, fname in enumerate(fnames):
        ess[fname] = pystan.chains.ess(sim_copy, fi)
    return ess

def get_sample_ess(fit):
    # Call stan built-in function to calculate ESS.
    sim_copy = copy.deepcopy(fit.sim)
    fnames = copy.copy(sim_copy['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    ess = {}
    for fi, fname in enumerate(fnames):
        ess[fname] = pystan.chains.ess(sim_copy, fi)
    return ess

def get_se_mean(sd, ess):
    fnames = list(sd.keys())
    se_means = {}
    for fname in fnames:
        se_means[fname] = sd[fname] / np.sqrt(ess[fname])
    return se_means

def stansummary_control_variates(fit, cv_samples, cv_samples_squared, digits_summary=2):
    parameters_mean, cv_sample_std = get_cv_sample_mean_std(fit, cv_samples)
    parameters_std = get_parameter_std(fit, cv_samples, cv_samples_squared)
    cv_sample_ess = get_cv_sample_ess(fit, cv_samples)
    mcmc_sample_ess = get_sample_ess(fit)
    se_mean = get_se_mean(cv_sample_std, cv_sample_ess)

    fnames = list(parameters_mean.keys())
    content = np.array([list(parameters_mean.values()), list(se_mean.values()), list(parameters_std.values()), list(cv_sample_ess.values())])
    body = _array_to_table(content.T, fnames, ['mean', 'se_mean', 'sd', 'n_eff'], digits_summary)
    
    return body


