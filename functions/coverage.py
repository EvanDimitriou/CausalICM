# packages
import numpy as np
import GPy
from sklearn.metrics import mean_squared_error
from functions.model_training import ICM
def coverage1(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 1 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)

    # True CATE
    CATE = 1+test_data

    # Create empty arrayes to store results
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))

    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation1
        data_exp, data_obs = data_simulation1(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


        # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95




def coverage2(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 2 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)
    CATE = 1+test_data+test_data**2
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation2
        data_exp, data_obs = data_simulation2(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


         # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95



def coverage3(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 3 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)
    CATE = 1+test_data
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation3
        data_exp, data_obs = data_simulation3(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


         # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95



def coverage4(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 4 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)
    CATE = 1+test_data+test_data**2
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation4
        data_exp, data_obs = data_simulation4(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


         # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95




###############################

def coverage5(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 5 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)
    CATE = 1+test_data
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation5
        data_exp, data_obs = data_simulation5(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


         # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95




def coverage6(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 6 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)
    CATE = 1+test_data+test_data**2
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation6
        data_exp, data_obs = data_simulation6(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


         # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95



def coverage7(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 7 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)
    CATE = 1+test_data
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation7
        data_exp, data_obs = data_simulation7(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


         # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95



def coverage8(num_datasets, min, max, step, sample_size, beta0, beta1, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 8 (univariate case)

        num_datasets: int
            the number of distinct datasets to iterate the experiment
        min: float
            The min value of the covariate in the test set
        max: float
            The max value of the covariate in the test set
        step: float
            Spacing between values in the test set
        sample_size: int
            Sample size of the observational study for the training sets. The trial sample size is a fraction of this number.
        beta0: float
            Intercept parameter of the logistic participation model.
        beta1: float
            Slope parameter of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.arange(min, max, step)
    CATE = 1+test_data+test_data**2
    coverage_ICM_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_ICM_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPobs_mat90 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat95 = np.zeros((test_data.shape[0], num_datasets))
    coverage_GPexp_mat90 = np.zeros((test_data.shape[0], num_datasets))
    length_ICM90 = np.zeros((test_data.shape[0]))
    length_ICM95 = np.zeros((test_data.shape[0]))
    length_GPobs90 = np.zeros((test_data.shape[0]))
    length_GPobs95 = np.zeros((test_data.shape[0]))
    length_GPexp90 = np.zeros((test_data.shape[0]))
    length_GPexp95 = np.zeros((test_data.shape[0]))
    mse_ICM = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation import data_simulation8
        data_exp, data_obs = data_simulation8(sample_size=sample_size, beta0=beta0, beta1=beta1)

        X_obs = data_obs[:,1]
        T_obs = data_obs[:,2]
        Y_obs = data_obs[:,3]
        X_exp = data_exp[:,1]
        T_exp = data_exp[:,2]
        Y_exp = data_exp[:,3]

        # Naïve GP - Observational

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==0]),np.vstack(Y_obs[T_obs==0]),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X_obs[T_obs==1]),np.vstack(Y_obs[T_obs==1]),kern)
        m1_obs.optimize()
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naïve GP - Experimental

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==0]),np.vstack(Y_exp[T_exp==0]),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X_exp[T_exp==1]),np.vstack(Y_exp[T_exp==1]),kern)
        m1_exp.optimize()
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)

        # ICM
        
        m0, m1 = ICM(X_E = data_exp[:,1], X_O = data_obs[:,1], 
                     T_E = data_exp[:,2], T_O = data_obs[:,2], 
                     Y_E = data_exp[:,3], Y_O = data_obs[:,3], 
                     r=r,ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data),np.ones(test_data.shape[0])*0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        mse_GPobs[i] = mean_squared_error(CATE_GPobs, CATE)
        mse_GPexp[i] = mean_squared_error(CATE_GPexp, CATE)
        mse_ICM[i] = mean_squared_error(predCATE_exp, CATE)
        # Compute the SD
        varCATE_exp = predY1_exp[1] + predY0_exp[1]
        sd_exp = np.sqrt(varCATE_exp)


         # Calculate credible intervals - ICM
        lower_bound_GPobs_95 = CATE_GPobs - 1.96*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_95 = CATE_GPobs + 1.96*np.vstack(sdCATE_GPobs)
        lower_bound_GPobs_90 = CATE_GPobs - 1.645*np.vstack(sdCATE_GPobs)
        upper_bound_GPobs_90 = CATE_GPobs + 1.645*np.vstack(sdCATE_GPobs)

         # Calculate credible intervals - ICM
        lower_bound_GPexp_95 = CATE_GPexp - 1.96*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_95 = CATE_GPexp + 1.96*np.vstack(sdCATE_GPexp)
        lower_bound_GPexp_90 = CATE_GPexp - 1.645*np.vstack(sdCATE_GPexp)
        upper_bound_GPexp_90 = CATE_GPexp + 1.645*np.vstack(sdCATE_GPexp)


        # Calculate credible intervals - ICM
        lower_bound_ICM_95 = predCATE_exp - 1.96*np.vstack(sd_exp)
        upper_bound_ICM_95 = predCATE_exp + 1.96*np.vstack(sd_exp)
        lower_bound_ICM_90 = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound_ICM_90 = predCATE_exp + 1.645*np.vstack(sd_exp)
        # Calculate length
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_95[iter]<=CATE[iter]<=upper_bound_ICM_95[iter]:
                coverage_ICM_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_ICM_90[iter]<=CATE[iter]<=upper_bound_ICM_90[iter]:
                coverage_ICM_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_95[iter]<=CATE[iter]<=upper_bound_GPobs_95[iter]:
                coverage_GPobs_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPobs_90[iter]<=CATE[iter]<=upper_bound_GPobs_90[iter]:
                coverage_GPobs_mat90[iter, i] = 1
        
        # Check if true parameter values are within the interval
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_95[iter]<=CATE[iter]<=upper_bound_GPexp_95[iter]:
                coverage_GPexp_mat95[iter, i] = 1
        
        for iter in range(test_data.shape[0]):
            if lower_bound_GPexp_90[iter]<=CATE[iter]<=upper_bound_GPexp_90[iter]:
                coverage_GPexp_mat90[iter, i] = 1
        
        # Length

        for iter in range(test_data.shape[0]):
            length_ICM90[iter] = upper_bound_ICM_90[iter] - lower_bound_ICM_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_ICM95[iter] = upper_bound_ICM_95[iter] - lower_bound_ICM_95[iter]

        for iter in range(test_data.shape[0]):
            length_GPobs90[iter] = upper_bound_GPobs_90[iter] - lower_bound_GPobs_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPobs95[iter] = upper_bound_GPobs_95[iter] - lower_bound_GPobs_95[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp90[iter] = upper_bound_GPexp_90[iter] - lower_bound_GPexp_90[iter]
        
        for iter in range(test_data.shape[0]):
            length_GPexp95[iter] = upper_bound_GPexp_95[iter] - lower_bound_GPexp_95[iter]
        
    # Calculate coverage probability
    coverage_prob_ICM_95 = np.mean(coverage_ICM_mat95, axis = 1)  
    coverage_prob_ICM_90 = np.mean(coverage_ICM_mat90, axis = 1) 

    coverage_prob_GPobs_95 = np.mean(coverage_GPobs_mat95, axis = 1)  
    coverage_prob_GPobs_90 = np.mean(coverage_GPobs_mat90, axis = 1) 

    coverage_prob_GPexp_95 = np.mean(coverage_GPexp_mat95, axis = 1)  
    coverage_prob_GPexp_90 = np.mean(coverage_GPexp_mat90, axis = 1)   
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_prob_ICM_95, coverage_prob_ICM_90, coverage_prob_GPobs_95, coverage_prob_GPobs_90, coverage_prob_GPexp_95, coverage_prob_GPexp_90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95
