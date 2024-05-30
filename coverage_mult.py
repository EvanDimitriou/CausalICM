# packages
import numpy as np
import GPy
from sklearn.metrics import mean_squared_error
from functions.model_training import ICM
def coverage1(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 1 (multivariate case)

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
            Slope parameter for X1 of the logistic participation model.
        beta2: float
            Slope parameter for X2 of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,1]

    coverage_ICM95 = np.zeros((num_datasets))
    coverage_ICM90 = np.zeros((num_datasets))
    length_ICM90 = np.zeros((num_datasets))
    length_ICM95 = np.zeros((num_datasets))
    mse_ICM = np.zeros((num_datasets))

    coverage_GPobs95 = np.zeros((num_datasets))
    coverage_GPobs90 = np.zeros((num_datasets))
    length_GPobs90 = np.zeros((num_datasets))
    length_GPobs95 = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))

    coverage_GPexp95 = np.zeros((num_datasets))
    coverage_GPexp90 = np.zeros((num_datasets))
    length_GPexp90 = np.zeros((num_datasets))
    length_GPexp95 = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))

    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation_multivariate import data_simulation1        
        data_exp, data_obs = data_simulation1(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

        # Data generation and preprocessing
        X0_obs = data_obs[:,1:6][data_obs[:,6]==0]
        X1_obs = data_obs[:,1:6][data_obs[:,6]==1]
        Y0_obs = data_obs[:,7][data_obs[:,6]==0]
        Y1_obs = data_obs[:,7][data_obs[:,6]==1]

        X0_exp = data_exp[:,1:6][data_exp[:,6]==0]
        X1_exp = data_exp[:,1:6][data_exp[:,6]==1]
        Y0_exp = data_exp[:,7][data_exp[:,6]==0]
        Y1_exp = data_exp[:,7][data_exp[:,6]==1]

        # Naive GP trained on the observational data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X0_obs),np.vstack(Y0_obs),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X1_obs),np.vstack(Y1_obs),kern)
        m1_obs.optimize()
        # Predict the expectation
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naive GP trained on the experimental data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X0_exp),np.vstack(Y0_exp),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X1_exp),np.vstack(Y1_exp),kern)
        m1_exp.optimize()
        # Predict the expectation
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)


        #Train model
        
        m0, m1 = ICM(X_E = data_exp[:,1:6], X_O = data_obs[:,1:6], 
                     T_E = data_exp[:,6], T_O = data_obs[:,6], 
                     Y_E = data_exp[:,7], Y_O = data_obs[:,7], 
                     r=r, ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        #mse[i] = mean_squared_error(predCATE_exp, CATE)
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
        
        # Check if true parameter values are within the interval
        
        if lower_bound_ICM_95<=CATE<=upper_bound_ICM_95:
            coverage_ICM95[i] = 1
        
        if lower_bound_ICM_90<=CATE<=upper_bound_ICM_90:
            coverage_ICM90[i] = 1
        
        if lower_bound_GPobs_95<=CATE<=upper_bound_GPobs_95:
            coverage_GPobs95[i] = 1
        
        if lower_bound_GPobs_90<=CATE<=upper_bound_GPobs_90:
            coverage_GPobs90[i] = 1

        if lower_bound_GPexp_95<=CATE<=upper_bound_GPexp_95:
            coverage_GPexp95[i] = 1
        
        if lower_bound_GPexp_90<=CATE<=upper_bound_GPexp_90:
            coverage_GPexp90[i] = 1
        
        mse_GPobs[i] = (CATE_GPobs - CATE)**2
        mse_GPexp[i] = (CATE_GPexp - CATE)**2
        mse_ICM[i] = (predCATE_exp - CATE)**2

        length_ICM95[i] = upper_bound_ICM_95 - lower_bound_ICM_95
        length_ICM90[i] = upper_bound_ICM_90 - lower_bound_ICM_90
        length_GPobs95[i] = upper_bound_GPobs_95 - lower_bound_GPobs_95
        length_GPobs90[i] = upper_bound_GPobs_90 - lower_bound_GPobs_90
        length_GPexp95[i] = upper_bound_GPexp_95 - lower_bound_GPexp_95
        length_GPexp90[i] = upper_bound_GPexp_90 - lower_bound_GPexp_90

    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_ICM95, coverage_ICM90, coverage_GPobs95, coverage_GPobs90, coverage_GPexp95, coverage_GPexp90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95

def coverage2(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 2 (multivariate case)

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
            Slope parameter for X1 of the logistic participation model.
        beta2: float
            Slope parameter for X2 of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,0]**2+test_data[0,1]+test_data[0,1]**2

    coverage_ICM95 = np.zeros((num_datasets))
    coverage_ICM90 = np.zeros((num_datasets))
    length_ICM90 = np.zeros((num_datasets))
    length_ICM95 = np.zeros((num_datasets))
    mse_ICM = np.zeros((num_datasets))

    coverage_GPobs95 = np.zeros((num_datasets))
    coverage_GPobs90 = np.zeros((num_datasets))
    length_GPobs90 = np.zeros((num_datasets))
    length_GPobs95 = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))

    coverage_GPexp95 = np.zeros((num_datasets))
    coverage_GPexp90 = np.zeros((num_datasets))
    length_GPexp90 = np.zeros((num_datasets))
    length_GPexp95 = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))

    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation_multivariate import data_simulation2      
        data_exp, data_obs = data_simulation2(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

        # Data generation and preprocessing
        X0_obs = data_obs[:,1:6][data_obs[:,6]==0]
        X1_obs = data_obs[:,1:6][data_obs[:,6]==1]
        Y0_obs = data_obs[:,7][data_obs[:,6]==0]
        Y1_obs = data_obs[:,7][data_obs[:,6]==1]

        X0_exp = data_exp[:,1:6][data_exp[:,6]==0]
        X1_exp = data_exp[:,1:6][data_exp[:,6]==1]
        Y0_exp = data_exp[:,7][data_exp[:,6]==0]
        Y1_exp = data_exp[:,7][data_exp[:,6]==1]

        # Naive GP trained on the observational data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X0_obs),np.vstack(Y0_obs),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X1_obs),np.vstack(Y1_obs),kern)
        m1_obs.optimize()
        # Predict the expectation
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naive GP trained on the experimental data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X0_exp),np.vstack(Y0_exp),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X1_exp),np.vstack(Y1_exp),kern)
        m1_exp.optimize()
        # Predict the expectation
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)


        #Train model
        
        m0, m1 = ICM(X_E = data_exp[:,1:6], X_O = data_obs[:,1:6], 
                     T_E = data_exp[:,6], T_O = data_obs[:,6], 
                     Y_E = data_exp[:,7], Y_O = data_obs[:,7], 
                     r=r, ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        #mse[i] = mean_squared_error(predCATE_exp, CATE)
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
        
        # Check if true parameter values are within the interval
        
        if lower_bound_ICM_95<=CATE<=upper_bound_ICM_95:
            coverage_ICM95[i] = 1
        
        if lower_bound_ICM_90<=CATE<=upper_bound_ICM_90:
            coverage_ICM90[i] = 1
        
        if lower_bound_GPobs_95<=CATE<=upper_bound_GPobs_95:
            coverage_GPobs95[i] = 1
        
        if lower_bound_GPobs_90<=CATE<=upper_bound_GPobs_90:
            coverage_GPobs90[i] = 1

        if lower_bound_GPexp_95<=CATE<=upper_bound_GPexp_95:
            coverage_GPexp95[i] = 1
        
        if lower_bound_GPexp_90<=CATE<=upper_bound_GPexp_90:
            coverage_GPexp90[i] = 1
        
        mse_GPobs[i] = (CATE_GPobs - CATE)**2
        mse_GPexp[i] = (CATE_GPexp - CATE)**2
        mse_ICM[i] = (predCATE_exp - CATE)**2

        length_ICM95[i] = upper_bound_ICM_95 - lower_bound_ICM_95
        length_ICM90[i] = upper_bound_ICM_90 - lower_bound_ICM_90
        length_GPobs95[i] = upper_bound_GPobs_95 - lower_bound_GPobs_95
        length_GPobs90[i] = upper_bound_GPobs_90 - lower_bound_GPobs_90
        length_GPexp95[i] = upper_bound_GPexp_95 - lower_bound_GPexp_95
        length_GPexp90[i] = upper_bound_GPexp_90 - lower_bound_GPexp_90


    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_ICM95, coverage_ICM90, coverage_GPobs95, coverage_GPobs90, coverage_GPexp95, coverage_GPexp90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95


def coverage3(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 3 (multivariate case)

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
            Slope parameter for X1 of the logistic participation model.
        beta2: float
            Slope parameter for X2 of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,1]

    coverage_ICM95 = np.zeros((num_datasets))
    coverage_ICM90 = np.zeros((num_datasets))
    length_ICM90 = np.zeros((num_datasets))
    length_ICM95 = np.zeros((num_datasets))
    mse_ICM = np.zeros((num_datasets))

    coverage_GPobs95 = np.zeros((num_datasets))
    coverage_GPobs90 = np.zeros((num_datasets))
    length_GPobs90 = np.zeros((num_datasets))
    length_GPobs95 = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))

    coverage_GPexp95 = np.zeros((num_datasets))
    coverage_GPexp90 = np.zeros((num_datasets))
    length_GPexp90 = np.zeros((num_datasets))
    length_GPexp95 = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))

    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation_multivariate import data_simulation3       
        data_exp, data_obs = data_simulation3(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

        # Data generation and preprocessing
        X0_obs = data_obs[:,1:6][data_obs[:,6]==0]
        X1_obs = data_obs[:,1:6][data_obs[:,6]==1]
        Y0_obs = data_obs[:,7][data_obs[:,6]==0]
        Y1_obs = data_obs[:,7][data_obs[:,6]==1]

        X0_exp = data_exp[:,1:6][data_exp[:,6]==0]
        X1_exp = data_exp[:,1:6][data_exp[:,6]==1]
        Y0_exp = data_exp[:,7][data_exp[:,6]==0]
        Y1_exp = data_exp[:,7][data_exp[:,6]==1]

        # Naive GP trained on the observational data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X0_obs),np.vstack(Y0_obs),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X1_obs),np.vstack(Y1_obs),kern)
        m1_obs.optimize()
        # Predict the expectation
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naive GP trained on the experimental data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X0_exp),np.vstack(Y0_exp),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X1_exp),np.vstack(Y1_exp),kern)
        m1_exp.optimize()
        # Predict the expectation
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)


        #Train model
        
        m0, m1 = ICM(X_E = data_exp[:,1:6], X_O = data_obs[:,1:6], 
                     T_E = data_exp[:,6], T_O = data_obs[:,6], 
                     Y_E = data_exp[:,7], Y_O = data_obs[:,7], 
                     r=r, ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        #mse[i] = mean_squared_error(predCATE_exp, CATE)
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
        
        # Check if true parameter values are within the interval
        
        if lower_bound_ICM_95<=CATE<=upper_bound_ICM_95:
            coverage_ICM95[i] = 1
        
        if lower_bound_ICM_90<=CATE<=upper_bound_ICM_90:
            coverage_ICM90[i] = 1
        
        if lower_bound_GPobs_95<=CATE<=upper_bound_GPobs_95:
            coverage_GPobs95[i] = 1
        
        if lower_bound_GPobs_90<=CATE<=upper_bound_GPobs_90:
            coverage_GPobs90[i] = 1

        if lower_bound_GPexp_95<=CATE<=upper_bound_GPexp_95:
            coverage_GPexp95[i] = 1
        
        if lower_bound_GPexp_90<=CATE<=upper_bound_GPexp_90:
            coverage_GPexp90[i] = 1
        
        mse_GPobs[i] = (CATE_GPobs - CATE)**2
        mse_GPexp[i] = (CATE_GPexp - CATE)**2
        mse_ICM[i] = (predCATE_exp - CATE)**2

        length_ICM95[i] = upper_bound_ICM_95 - lower_bound_ICM_95
        length_ICM90[i] = upper_bound_ICM_90 - lower_bound_ICM_90
        length_GPobs95[i] = upper_bound_GPobs_95 - lower_bound_GPobs_95
        length_GPobs90[i] = upper_bound_GPobs_90 - lower_bound_GPobs_90
        length_GPexp95[i] = upper_bound_GPexp_95 - lower_bound_GPexp_95
        length_GPexp90[i] = upper_bound_GPexp_90 - lower_bound_GPexp_90
    
    # Calculate length
    return mse_ICM, mse_GPobs, mse_GPexp, coverage_ICM95, coverage_ICM90, coverage_GPobs95, coverage_GPobs90, coverage_GPexp95, coverage_GPexp90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95

def coverage4(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, 90% and 95% coverage and interval width for Simulation 1 (multivariate case)

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
            Slope parameter for X1 of the logistic participation model.
        beta2: float
            Slope parameter for X2 of the logistic participation model.
        r: int
            Rank of the ICM.
        ID: int
            input dim for ICM.
        AD: list)
            active dim for ICM
        rho: trust hyperparameter of ICM
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,0]**2+test_data[0,1]+test_data[0,1]**2

    coverage_ICM95 = np.zeros((num_datasets))
    coverage_ICM90 = np.zeros((num_datasets))
    length_ICM90 = np.zeros((num_datasets))
    length_ICM95 = np.zeros((num_datasets))
    mse_ICM = np.zeros((num_datasets))

    coverage_GPobs95 = np.zeros((num_datasets))
    coverage_GPobs90 = np.zeros((num_datasets))
    length_GPobs90 = np.zeros((num_datasets))
    length_GPobs95 = np.zeros((num_datasets))
    mse_GPobs = np.zeros((num_datasets))

    coverage_GPexp95 = np.zeros((num_datasets))
    coverage_GPexp90 = np.zeros((num_datasets))
    length_GPexp90 = np.zeros((num_datasets))
    length_GPexp95 = np.zeros((num_datasets))
    mse_GPexp = np.zeros((num_datasets))

    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from functions.data_simulation_multivariate import data_simulation4     
        data_exp, data_obs = data_simulation4(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

        # Data generation and preprocessing
        X0_obs = data_obs[:,1:6][data_obs[:,6]==0]
        X1_obs = data_obs[:,1:6][data_obs[:,6]==1]
        Y0_obs = data_obs[:,7][data_obs[:,6]==0]
        Y1_obs = data_obs[:,7][data_obs[:,6]==1]

        X0_exp = data_exp[:,1:6][data_exp[:,6]==0]
        X1_exp = data_exp[:,1:6][data_exp[:,6]==1]
        Y0_exp = data_exp[:,7][data_exp[:,6]==0]
        Y1_exp = data_exp[:,7][data_exp[:,6]==1]

        # Naive GP trained on the observational data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_obs = GPy.models.GPRegression(np.vstack(X0_obs),np.vstack(Y0_obs),kern)
        m0_obs.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_obs = GPy.models.GPRegression(np.vstack(X1_obs),np.vstack(Y1_obs),kern)
        m1_obs.optimize()
        # Predict the expectation
        mu0_obs = m0_obs.predict(np.vstack(test_data))[0]
        mu1_obs = m1_obs.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPobs = mu1_obs - mu0_obs
        varCATE_GPobs = m0_obs.predict_noiseless(np.vstack(test_data))[1] + m1_obs.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPobs = np.sqrt(varCATE_GPobs)

        # Naive GP trained on the experimental data
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m0_exp = GPy.models.GPRegression(np.vstack(X0_exp),np.vstack(Y0_exp),kern)
        m0_exp.optimize()
        kern = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
        m1_exp = GPy.models.GPRegression(np.vstack(X1_exp),np.vstack(Y1_exp),kern)
        m1_exp.optimize()
        # Predict the expectation
        mu0_exp = m0_exp.predict(np.vstack(test_data))[0]
        mu1_exp = m1_exp.predict(np.vstack(test_data))[0]
        # Derive CATE
        CATE_GPexp = mu1_exp - mu0_exp
        varCATE_GPexp = m0_exp.predict_noiseless(np.vstack(test_data))[1] + m1_exp.predict_noiseless(np.vstack(test_data))[1] 
        sdCATE_GPexp = np.sqrt(varCATE_GPexp)


        #Train model
        
        m0, m1 = ICM(X_E = data_exp[:,1:6], X_O = data_obs[:,1:6], 
                     T_E = data_exp[:,6], T_O = data_obs[:,6], 
                     Y_E = data_exp[:,7], Y_O = data_obs[:,7], 
                     r=r, ID = ID, AD = AD, rho = rho)
        # Compute the CATE
        predY0_exp = m0.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predY1_exp = m1.predict_noiseless(np.c_[np.vstack(test_data), np.ones(test_data.shape[0]) * 0])
        predCATE_exp = predY1_exp[0] - predY0_exp[0]

        #mse[i] = mean_squared_error(predCATE_exp, CATE)
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
        
        # Check if true parameter values are within the interval
        
        if lower_bound_ICM_95<=CATE<=upper_bound_ICM_95:
            coverage_ICM95[i] = 1
        
        if lower_bound_ICM_90<=CATE<=upper_bound_ICM_90:
            coverage_ICM90[i] = 1
        
        if lower_bound_GPobs_95<=CATE<=upper_bound_GPobs_95:
            coverage_GPobs95[i] = 1
        
        if lower_bound_GPobs_90<=CATE<=upper_bound_GPobs_90:
            coverage_GPobs90[i] = 1

        if lower_bound_GPexp_95<=CATE<=upper_bound_GPexp_95:
            coverage_GPexp95[i] = 1
        
        if lower_bound_GPexp_90<=CATE<=upper_bound_GPexp_90:
            coverage_GPexp90[i] = 1
        
        mse_GPobs[i] = (CATE_GPobs - CATE)**2
        mse_GPexp[i] = (CATE_GPexp - CATE)**2
        mse_ICM[i] = (predCATE_exp - CATE)**2

        length_ICM95[i] = upper_bound_ICM_95 - lower_bound_ICM_95
        length_ICM90[i] = upper_bound_ICM_90 - lower_bound_ICM_90
        length_GPobs95[i] = upper_bound_GPobs_95 - lower_bound_GPobs_95
        length_GPobs90[i] = upper_bound_GPobs_90 - lower_bound_GPobs_90
        length_GPexp95[i] = upper_bound_GPexp_95 - lower_bound_GPexp_95
        length_GPexp90[i] = upper_bound_GPexp_90 - lower_bound_GPexp_90

    return mse_ICM, mse_GPobs, mse_GPexp, coverage_ICM95, coverage_ICM90, coverage_GPobs95, coverage_GPobs90, coverage_GPexp95, coverage_GPexp90, length_ICM90, length_ICM95, length_GPobs90, length_GPobs95, length_GPexp90, length_GPexp95
