import pystan
import numpy as np
import rpy2.robjects as robjects
import pystan.postprocess as pp
from os import path
from examples.utils import verifyDataType, getParameterNames


if __name__ == '__main__':
    with open('examples/list_of_models', 'r') as fid:
        file_dirs = fid.readlines()
    
    for file_dir in file_dirs:
        file_dir = file_dir.strip()
        print(file_dir)
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
            fit = sm.sampling(data=data, chains=2, iter=500, warmup=300, verbose=True, algorithm='Fixed_param')
        else:
            fit = sm.sampling(data=data, chains=2, iter=500, warmup=300, verbose=True)

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

