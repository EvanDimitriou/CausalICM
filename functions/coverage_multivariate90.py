# packages
import numpy as np
from sklearn.metrics import mean_squared_error
from neurIPS.functions.model_training import ICM
def coverage1_90(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, coverage and interval width when RCT covariate support and observational covariate support are the same
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,1]
    coverage = np.zeros((num_datasets))
    mse = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from neurIPS.functions.data_simulation_multivariate import data_simulation1        
        data_exp, data_obs = data_simulation1(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

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

        # Calculate credible intervals
        lower_bound = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound = predCATE_exp + 1.645*np.vstack(sd_exp)
        
        # Check if true parameter values are within the interval
        
        if lower_bound<=CATE<=upper_bound:
            coverage[i] = 1
        
    # Calculate coverage probability
    coverage_prob = np.mean(coverage)   
    
    # Calculate length
    return coverage_prob

def coverage2_90(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, coverage and interval width when RCT covariate support and observational covariate support are the same
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,1]+test_data[0,0]**2+test_data[0,1]**2
    coverage = np.zeros((num_datasets))
    mse = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from neurIPS.functions.data_simulation_multivariate import data_simulation2        
        data_exp, data_obs = data_simulation2(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

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

        # Calculate credible intervals
        lower_bound = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound = predCATE_exp + 1.645*np.vstack(sd_exp)
        
        # Check if true parameter values are within the interval
        
        if lower_bound<=CATE<=upper_bound:
            coverage[i] = 1
        
    # Calculate coverage probability
    coverage_prob = np.mean(coverage)   
    
    # Calculate length
    return coverage_prob

def coverage3_90(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, coverage and interval width when RCT covariate support and observational covariate support are the same
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,1]
    coverage = np.zeros((num_datasets))
    mse = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from neurIPS.functions.data_simulation_multivariate import data_simulation3        
        data_exp, data_obs = data_simulation3(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

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

        # Calculate credible intervals
        lower_bound = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound = predCATE_exp + 1.645*np.vstack(sd_exp)
        
        # Check if true parameter values are within the interval
        
        if lower_bound<=CATE<=upper_bound:
            coverage[i] = 1
        
    # Calculate coverage probability
    coverage_prob = np.mean(coverage)   
    
    # Calculate length
    return coverage_prob

def coverage4_90(num_datasets, sample_size, beta0, beta1, beta2, r, ID, AD, rho):
    """
    MSE, coverage and interval width when RCT covariate support and observational covariate support are the same
    """
    # create test dataset
    test_data = np.c_[np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1),
                      np.random.uniform(-2, 2, 1)]
    CATE = 1+test_data[0,0]+test_data[0,1]+test_data[0,0]**2+test_data[0,1]**2
    coverage = np.zeros((num_datasets))
    mse = np.zeros((num_datasets))
    for i in range(num_datasets):
        print(i+1, end=",")
        # Generate data 
        from neurIPS.functions.data_simulation_multivariate import data_simulation4       
        data_exp, data_obs = data_simulation4(sample_size=sample_size, beta0=beta0, beta1=beta1, beta2 = beta2)

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

        # Calculate credible intervals
        lower_bound = predCATE_exp - 1.645*np.vstack(sd_exp)
        upper_bound = predCATE_exp + 1.645*np.vstack(sd_exp)
        
        # Check if true parameter values are within the interval
        
        if lower_bound<=CATE<=upper_bound:
            coverage[i] = 1
        
    # Calculate coverage probability
    coverage_prob = np.mean(coverage)   
    
    # Calculate length
    return coverage_prob