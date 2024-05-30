import GPy
import numpy as np

def ICM(X_E, X_O, T_E, T_O, Y_E, Y_O, r, ID, AD, rho):
    """
    Train and return two rank-2 ICMs - one for the control arm and one for the treated arm 
    Args:
    - X_E (numpy.ndarray): Baseline covariate in the experimental data.
    - X_O (numpy.ndarray): Baseline covariate in the observational data.
    - T_E (numpy.ndarray): Treatment indicator in the experimental data.
    - T_O (numpy.ndarray): Treatment indicator in the observational data.
    - Y_E (numpy.ndarray): Outcome in the experimental data.
    - Y_O (numpy.ndarray): Outcome in the observational data.
    - r (int): Rank of the model.
    - ID (int): input dim
    - AD (list): active dim
    - rho: trust hyperparameter

    Returns:
    GPy.models.GPRegression: Trained Gaussian Process Regression model using Intrinsic Coregionalized Model (ICM).

    Data Processing:
    - Split the variables into treated and control for both observational and experimental data.

    Model Training:
    - Intrinsic Coregionalized Model (ICM) with a Radial Basis Function (RBF) kernel is used for regression.
    - The model is trained using Gaussian Process Regression.

    Note:
    - The order of arranging data matters: Experimental control, Observational control, Experimental treated, Observational treated.
    - An indicator is added to the baseline covariate to represent the task (0 for experimental control, 1 for observational control, 2 for experimental treated, 3 for observational treated).
    """

    ##############################
    ##### Data preprocessing #####
    ##############################

    # Split the variables in treated and control 
    # Observational Data
    X0_obs = X_O[T_O==0] # baseline covariate in th control arm of the observational study
    X1_obs = X_O[T_O==1] # baseline covariate in th treatment arm of the observational study
    Y0_obs = Y_O[T_O==0] # outcomee in th control arm of the observational study
    Y1_obs = Y_O[T_O==1] # outcomee in th treatment arm of the observational study
    
    #Exoerimental data
    X0_exp = X_E[T_E==0] # baseline covariate in th control arm of the experimental study
    X1_exp = X_E[T_E==1] # baseline covariate in th treatment arm of the experimental study
    Y0_exp = Y_E[T_E==0] # outcomee in th control arm of the experimental study
    Y1_exp = Y_E[T_E==1] # outcomee in th treatment arm of the experimental study
    

    # The order we arrange the data matters
    # 1. Experimental control
    # 2. Observational control
    # 3. Experimental treated
    # 4. Observational treated
    # We also need to add an indicator to the baseline covariate indicating the task
    # i.e. 0 is for the experiemental control, 1 is for the observational control,
    # 2 is for the experiemental treated, 3 is for observational treated
    X0_exp = np.c_[X0_exp,np.ones(X0_exp.shape[0])*0] 
    X0_obs = np.c_[X0_obs,np.ones(X0_obs.shape[0])*1]
    X1_exp = np.c_[X1_exp,np.ones(X1_exp.shape[0])*0]
    X1_obs = np.c_[X1_obs,np.ones(X1_obs.shape[0])*1]

    # Put all the data on a single array 
    Y0 = np.r_[Y0_exp, Y0_obs]
    X0 = np.r_[X0_exp, X0_obs]
    Y1 = np.r_[Y1_exp, Y1_obs]
    X1 = np.r_[X1_exp, X1_obs]

    #########################################
    ###### Model Training - Control Arm #####
    #########################################
    
    # Define the kernel
    kern = GPy.kern.RBF(input_dim=ID, active_dims = AD)**GPy.kern.Coregionalize(input_dim = 1, output_dim = 2, rank=r)
    # Define the model
    m0 = GPy.models.GPRegression(X0,np.vstack(Y0),kern)
    ##############################################################
    # Put all your constraines here
    m0.mul.coregion.W[0,0:1].fix(1)
    m0.mul.coregion.W[0,1:].fix(0)
    m0.mul.coregion.W[1,0:1].fix(rho)
    m0.mul.coregion.W[1,1:].fix(np.sqrt(1-rho**2))
    m0.mul.coregion.kappa.fix([0.000000001, 0.000000001])    
    # Optimize the model
    m0.optimize()

    #########################################
    ###### Model Training - Treated Arm #####
    #########################################

    # Define the kernel
    kern = GPy.kern.RBF(input_dim=ID, active_dims = AD)**GPy.kern.Coregionalize(input_dim = 1, output_dim = 2, rank=r)
    # Define the model
    m1 = GPy.models.GPRegression(X1,np.vstack(Y1),kern)
    ##############################################################
    # Put all your constraines here
    m1.mul.coregion.W[0,0:1].fix(1)
    m1.mul.coregion.W[0,1:].fix(0)
    m1.mul.coregion.W[1,0:1].fix(rho)
    m1.mul.coregion.W[1,1:].fix(np.sqrt(1-rho**2))
    m1.mul.coregion.kappa.fix([0.000000001, 0.000000001])
    # Optimize the model
    m1.optimize()

    return m0, m1