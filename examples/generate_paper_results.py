import copy
import time
import pystan
import numpy as np
import rpy2.robjects as robjects
import pystan.postprocess_for_paper as pp
import pandas as pd
import pickle
from os import path
from examples.utils import verifyDataType, getParameterNames


if __name__ == '__main__':
    with open('examples/list_of_models', 'r') as fid:
        file_dirs = fid.readlines()
    
    model_names = []
    all_param_count = []
    constrained_param_count = []
    rmse_mcmc_all = []
    rmse_lcv_all = []
    rmse_qcv_all = []

    vars_lcv_improv_all = []
    vars_qcv_improv_all = []
    vars_lcv_improv_cons_all = []
    vars_qcv_improv_cons_all = []

    build_time_all = []
    sample_time_all = []

    lcv_grad_time_all = []
    lcv_cv_time_all = []
    qcv_grad_time_all = []
    qcv_cv_time_all = []
    for iter, file_dir in enumerate(file_dirs):
        file_dir = file_dir.strip()
        print(file_dir)
        # Assume the stan model file ends with .stan
        model_file = file_dir + '.stan'
        # Assume the data file ends with .data.R
        data_file = file_dir + '.data.R'
        model_names.append(path.basename(file_dir))
        
        ## - - - - - Read R data file - - - - -
        robjects.globalenv.clear()
        # read data into env
        if path.exists(data_file):
            robjects.r['source'](data_file)
            # variables
            vars = list(robjects.globalenv.keys())
            if len(vars) > 0:
                data = {}
                for var in vars:
                    data_ = np.array(robjects.globalenv.find(var))
                    if (data_.ndim == 1) and (data_.shape[0] == 1):
                        data[var] = data_[0]
                    else:
                        data[var] = data_
            else:
                data = None
        else:
            data = None
        ## - - - - - - - - - - - - - - - - - - - - - - - -
        
        # run stanlinear_cv_samples
        build_start_time = time.time()
        sm = pystan.StanModel(file=model_file)
        build_time = time.time() - build_start_time
        parameter_only_names = getParameterNames(sm)

        if data is not None:
            data = verifyDataType(sm, data) # make sure the types of input data align with .stan file. 
        
        # prepare truth - - - - - - - - - - - - - - - - - - 
        if path.exists('/home/yifan/control-variate-paper-data/'+path.basename(file_dir)+'_truth.bin'):
            with open('/home/yifan/control-variate-paper-data/'+path.basename(file_dir)+'_truth.bin', 'rb') as fid:
                parameter_extract_truth = pickle.load(fid)
        else:
            fit_truth_prep = sm.sampling(data=data, chains=1, iter=200, warmup=100, verbose=False)
            parameter_names = copy.copy(fit_truth_prep.sim['pars_oi'])
            parameter_names.remove('lp__')
            fit_truth = sm.sampling(data=data, chains=1, iter=200000, warmup=100000, verbose=False)
            parameter_extract_truth = fit_truth.extract(parameter_names, permuted=False)
            with open('/home/yifan/control-variate-paper-data/'+path.basename(file_dir)+'_truth.bin', 'wb') as fid:
                pickle.dump(parameter_extract_truth, fid)
        # - - - - - - - - - - - - - - - - - - 

        # Monte carlo runs - - - - - - - - - - - - - - - - - - 
        rmse_mcmc_mc = []
        rmse_lcv_mc = []
        rmse_qcv_mc = []

        vars_lcv_improv_mc = []
        vars_qcv_improv_mc = []
        vars_lcv_improv_cons_mc = []
        vars_qcv_improv_cons_mc = []

        vars_all_mcmc_mc = []
        vars_all_lcv_mc = []
        vars_all_qcv_mc = []

        vars_all_con_mcmc_mc = []
        vars_all_con_lcv_mc = []
        vars_all_con_qcv_mc = []

        sample_time_mc = []
        lcv_grad_time_mc = []
        lcv_cv_time_mc = []
        qcv_grad_time_mc = []
        qcv_cv_time_mc = []

        for mc_run in range(100):
            sample_start_time = time.time()
            fit = sm.sampling(data=data, chains=1, iter=2000, warmup=1000, verbose=False)
            sample_time = time.time() - sample_start_time

            # print(fit.stansummary(digits_summary=5))
            print('Monte carlo run iter {:d}.'.format(mc_run + 1))
            # run control variates
            lcv_grad_times = []
            lcv_cv_times = []
            qcv_grad_times = []
            qcv_cv_times = []
            # calculate the average runtime, as it is too fast.
            linear_cv_samples, linear_cv_runtime, lcv_constrained_dim = pp.run_postprocess(fit, cv_mode='linear')
            quadratic_cv_samples, quadratic_cv_runtime, qcv_constrained_dim = pp.run_postprocess(fit, cv_mode='quadratic')
            lcv_grad_time_mean = linear_cv_runtime[0]
            lcv_cv_time_mean = linear_cv_runtime[1]
            qcv_grad_time_mean = quadratic_cv_runtime[0]
            qcv_cv_time_mean = quadratic_cv_runtime[1]

            # (Approximated) truth
            # fit_truth = sm.sampling(data=data, chains=1, iter=200000, warmup=100000, verbose=False)
            parameter_names = copy.copy(fit.sim['pars_oi'])
            parameter_names.remove('lp__')
            
            parameter_means_truth = {}
            parameter_means_vector_truth = []
            for pn in parameter_names:
                parameter_means_truth[pn] = np.mean(parameter_extract_truth[pn][:, 0], axis=0)
                if parameter_means_truth[pn].ndim == 0:
                    parameter_means_vector_truth.extend([parameter_means_truth[pn]])
                else:
                    parameter_means_vector_truth.extend(list(np.squeeze(parameter_means_truth[pn]).flatten(order='F')))
            parameter_means_vector_truth = np.array(parameter_means_vector_truth)

            # MCMC correctness
            parameter_extract = fit.extract(parameter_names, permuted=False)
            parameter_means = {}
            parameter_vars = {}
            parameter_means_vector = []
            parameter_vars_vector = []
            for pn in parameter_names:
                parameter_means[pn] = np.mean(parameter_extract[pn][:, 0], axis=0)
                parameter_vars[pn] = np.var(parameter_extract[pn][:, 0], axis=0)
                if parameter_means[pn].ndim == 0:
                    parameter_means_vector.extend([parameter_means[pn]])
                    parameter_vars_vector.extend([parameter_vars[pn]])
                else:
                    parameter_means_vector.extend(list(np.squeeze(parameter_means[pn]).flatten(order='F')))
                    parameter_vars_vector.extend(list(np.squeeze(parameter_vars[pn]).flatten(order='F')))
            parameter_means_vector = np.array(parameter_means_vector)
            parameter_vars_vector = np.array(parameter_vars_vector)
            rmse_mcmc = np.sqrt(np.nanmean((parameter_means_vector-parameter_means_vector_truth)**2))
            if len(lcv_constrained_dim) == 0:
                vars_cons_mcmc = np.nan
            else:
                vars_cons_mcmc = parameter_vars_vector[lcv_constrained_dim]

            # Linear CV correctness
            lcv_means = np.nanmean(linear_cv_samples, axis=0)
            rmse_lcv = np.sqrt(np.nanmean((lcv_means-parameter_means_vector_truth)**2))
            vars_lcv_tmp = np.nanvar(linear_cv_samples, axis=0)
            vars_lcv_improv = np.nanmean(parameter_vars_vector/vars_lcv_tmp)
            if len(lcv_constrained_dim) == 0:
                vars_cons_lcv = np.nan
            else:
                vars_cons_lcv = vars_lcv_tmp[lcv_constrained_dim]
            vars_lcv_improv_cons = np.nanmean(vars_cons_mcmc/vars_cons_lcv)

            # Quad CV correctness
            qcv_means = np.nanmean(quadratic_cv_samples, axis=0)
            rmse_qcv = np.sqrt(np.nanmean((qcv_means-parameter_means_vector_truth)**2))
            vars_qcv_tmp = np.nanvar(quadratic_cv_samples, axis=0)
            vars_qcv_improv = np.nanmean(parameter_vars_vector/vars_qcv_tmp)
            if len(lcv_constrained_dim) == 0:
                vars_cons_qcv = np.nan
            else:
                vars_cons_qcv = vars_qcv_tmp[lcv_constrained_dim]
            vars_qcv_improv_cons = np.nanmean(vars_cons_mcmc/vars_cons_qcv)

            sample_time_mc.append(sample_time)
            lcv_grad_time_mc.append(lcv_grad_time_mean)
            lcv_cv_time_mc.append(lcv_cv_time_mean)
            qcv_grad_time_mc.append(qcv_grad_time_mean)
            qcv_cv_time_mc.append(qcv_cv_time_mean)

            rmse_mcmc_mc.append(rmse_mcmc)
            rmse_lcv_mc.append(rmse_lcv)
            rmse_qcv_mc.append(rmse_qcv)

            vars_lcv_improv_mc.append(vars_lcv_improv)
            vars_qcv_improv_mc.append(vars_qcv_improv)
            vars_lcv_improv_cons_mc.append(vars_lcv_improv_cons)
            vars_qcv_improv_cons_mc.append(vars_qcv_improv_cons)

            vars_all_mcmc_mc.append(parameter_vars_vector)
            vars_all_lcv_mc.append(vars_lcv_tmp)
            vars_all_qcv_mc.append(vars_qcv_tmp)

            vars_all_con_mcmc_mc.append(vars_cons_mcmc)
            vars_all_con_lcv_mc.append(vars_cons_lcv)
            vars_all_con_qcv_mc.append(vars_cons_qcv)
            
        
        with open('/home/yifan/control-variate-paper-data/'+path.basename(file_dir)+'_variance_records.bin', 'wb') as fid:
            pickle.dump({'vars_mcmc': vars_all_mcmc_mc, 'vars_lcv':vars_all_lcv_mc, 'vars_qcv':vars_all_qcv_mc,
            'vars_all_con_mcmc_mc':vars_all_con_mcmc_mc, 'vars_all_con_lcv_mc':vars_all_con_lcv_mc, 'vars_all_con_qcv_mc':vars_all_con_qcv_mc}, fid)

        all_param_count.append(linear_cv_samples.shape[1])
        constrained_param_count.append(len(lcv_constrained_dim))
        
        build_time_all.append(build_time)
        sample_time_all.append(np.mean(sample_time_mc))
        lcv_grad_time_all.append(np.mean(lcv_grad_time_mc))
        lcv_cv_time_all.append(np.mean(lcv_cv_time_mc))
        qcv_grad_time_all.append(np.mean(qcv_grad_time_mc))
        qcv_cv_time_all.append(np.mean(qcv_cv_time_mc))

        rmse_mcmc_all.append(np.nanmean(rmse_mcmc_mc))
        rmse_lcv_all.append(np.nanmean(rmse_lcv_mc))
        rmse_qcv_all.append(np.nanmean(rmse_qcv_mc))

        vars_lcv_improv_all.append(np.nanmean(vars_lcv_improv_mc))
        vars_qcv_improv_all.append(np.nanmean(vars_qcv_improv_mc))
        vars_lcv_improv_cons_all.append(np.nanmean(vars_lcv_improv_cons_mc))
        vars_qcv_improv_cons_all.append(np.nanmean(vars_qcv_improv_cons_mc))

        paper_summary = {'model_name':model_names, 'all_params_count':all_param_count, 'constrained_param_count': constrained_param_count,
        'build_time': build_time_all, 'sample_time': sample_time_all, 
        'lcv_gradient_time':lcv_grad_time_all, 'lcv_cv_time':lcv_cv_time_all,
        'qcv_gradient_time':qcv_grad_time_all, 'qcv_cv_time':qcv_cv_time_all,
        'rmse_mcmc':rmse_mcmc_all, 'rmse_lcv':rmse_lcv_all, 'rmse_qcv':rmse_qcv_all,
        'vars_lcv_improv':vars_lcv_improv_all, 'vars_qcv_improv':vars_qcv_improv_all,
        'vars_lcv_improv_cons':vars_lcv_improv_cons_all, 'vars_qcv_improv_cons':vars_qcv_improv_cons_all}
        paper_summary_df = pd.DataFrame(paper_summary)
        paper_summary_df.to_csv('paper_summary_{:d}.csv'.format(iter), index=False)
        