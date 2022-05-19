import argparse
import pystan
import numpy as np
import pystan.postprocess as pp
import rpy2.robjects as robjects
from os import path
from examples.utils import verifyDataType, getParameterNames


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test stan models and postprocess samples using control variates.')
    parser.add_argument('--model', default='eight_schools', type=str, help='Model name.')
    args = parser.parse_args()

    if args.model == 'eight_schools':
        file_dir = './examples/models/eight_schools/eight_schools'
    elif args.model == 'low_dim_corr_gauss':
        file_dir = './examples/models/low_dim_corr_gauss/low_dim_corr_gauss'
    elif args.model == 'low_dim_gauss_mix':
        file_dir = './examples/models/low_dim_gauss_mix/low_dim_gauss_mix'
    elif args.model == 'arK':
        file_dir = './examples/models/arK/arK'
    elif args.model == 'arma':
        file_dir = './examples/models/arma/arma'
    elif args.model == 'garch':
        file_dir = './examples/models/garch/garch'
    elif args.model == 'gp_pois_regr':
        file_dir = './examples/models/gp_pois_regr/gp_pois_regr'
    elif args.model == 'gp_regr':
        file_dir = './examples/models/gp_regr/gp_regr'
    elif args.model == 'irt_2pl':
        file_dir = './examples/models/irt_2pl/irt_2pl'
    elif args.model == 'low_dim_gauss_mix_collapse':
        file_dir = './examples/models/low_dim_gauss_mix_collapse/low_dim_gauss_mix_collapse'
    elif args.model == 'one_comp_mm_elim_abs':
        file_dir = './examples/models/pkpd/one_comp_mm_elim_abs'
    elif args.model == 'sir':
        file_dir = './examples/models/sir/sir'
    else:
        file_dir = None
        print('Model name unknown...')

    if file_dir:
        print('Runing model {}'.format(args.model))
        # file_dir = './examples/models/arK/arK'
        # Assume the stan model file ends with .stan
        model_file = file_dir + '.stan'
        # Assume the data file ends with .data.R
        data_file = file_dir + '.data.R'

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
        
        # run stan
        sm = pystan.StanModel(file=model_file)
        parameter_names = getParameterNames(sm)

        if data is not None:
            data = verifyDataType(sm, data) # make sure the types of input data align with .stan file. 
        if parameter_names is None:
            fit = sm.sampling(data=data, chains=3, iter=2000, warmup=1000, verbose=True, algorithm='Fixed_param')
        else:
            fit = sm.sampling(data=data, chains=3, iter=2000, warmup=1000, verbose=True)

        print(fit.stansummary(digits_summary=5))
        # run control variates
        linear_cv_samples, linear_cv_samples_suqared = pp.run_postprocess(fit, output_squared_samples=True, cv_mode='linear')
        quadratic_cv_samples, quadratic_cv_samples_suqared = pp.run_postprocess(fit, output_squared_samples=True, cv_mode='quadratic')
        print('- - - - - - - - - - Control Variates - - - - - - - - - - ')
        lcv_stansummary_body = pp.stansummary_control_variates(fit, linear_cv_samples, linear_cv_samples_suqared, digits_summary=5)
        print(lcv_stansummary_body)
        print('- - - - - - - - - - Quadratic Control Variates - - - - - - - - - - ')
        qcv_stansummary_body = pp.stansummary_control_variates(fit, quadratic_cv_samples, quadratic_cv_samples_suqared, digits_summary=5)
        print(qcv_stansummary_body)
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - ')

