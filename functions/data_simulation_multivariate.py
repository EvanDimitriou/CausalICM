import numpy as np
import pandas as pd
from scipy.stats import bernoulli, norm

def data_simulation1(sample_size, beta0, beta1, beta2):
    """
    Simulate observational and trial data according to a specified logistic participation model.
    
        Confounding: Simple
        Y(0): Complex 
        Y(1): Complex
        CATE: Simple

    Parameters:
    sample_size : int
        Number of observational samples to generate. The trial sample size will be a fraction of that sample size
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariates (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

    ######################
    ##### Trial data #####
    ######################

    # Baseline covatiates 
    X1 = np.random.uniform(-2, 2, sample_size)
    X2 = np.random.uniform(-2, 2, sample_size)
    X3 = np.random.uniform(-2, 2, sample_size)
    X4 = np.random.uniform(-2, 2, sample_size)
    X5 = np.random.uniform(-2, 2, sample_size)
    
    # Probabilitiy of participation - affected by X1 and X2
    p_S = (np.exp(beta0 + beta1*X1 + beta2*X2)) / (1 + np.exp(beta0 + beta1*X1 + beta2*X2)) 

    # Study indicator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X1, X2, X3, X4, X5))
    S_trial = data[data[:,0]==1, 0]
    X_trial = np.c_[data[data[:,0]==1,1],
                    data[data[:,0]==1,2],
                    data[data[:,0]==1,3],
                    data[data[:,0]==1,4],
                    data[data[:,0]==1,5]]
    
    # Random treatment 
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE - a function of X1 and X2
    tau_X_trial = 1 + X_trial[:,0] + X_trial[:,1]

    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + (X_trial[:, 0]**2 + X_trial[:, 1]**2 + X_trial[:,2]**2 + X_trial[:,3]**2 + X_trial[:,4]**2) + epsilon_trial

    ##############################
    ##### Observationla data #####
    ##############################

    # Baseline covariates 
    X1_observational = np.random.uniform(-2, 2, sample_size)
    X2_observational = np.random.uniform(-2, 2, sample_size)
    X3_observational = np.random.uniform(-2, 2, sample_size)
    X4_observational = np.random.uniform(-2, 2, sample_size)
    X5_observational = np.random.uniform(-2, 2, sample_size)
    X_observational = np.c_[X1_observational, X2_observational, X3_observational,
                            X4_observational, X5_observational]
    
    # Study indicator
    S_observational = np.zeros(X1_observational.shape[0])

    # Biased treatment - a function of X1 and X2
    logit_e_X_0 = (X1_observational + X2_observational)
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X1_observational)[0])

    # CATE - a function of X1 and X2
    tau_X_observational = 1 + X1_observational + X2_observational

    # Unobserved confounder  - a function of X1 and X2
    U_observational = (2 * A_observational - 1) * ((X1_observational + X2_observational)) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X1_observational**2 + X2_observational**2 + X3_observational**2 + X4_observational**2 + X5_observational**2 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation


def data_simulation2(sample_size, beta0, beta1, beta2):

    """
    Simulate observational and trial data according to a specified logistic participation model.
    
        Confounding: Simple
        Y(0): Complex 
        Y(1): Complex
        CATE: Complex

    Parameters:
    sample_size : int
        Number of observational samples to generate. The trial sample size will be a fraction of that sample size
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariates (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

    ######################
    ##### Trial data #####
    ######################

    # Baseline covatiates 
    X1 = np.random.uniform(-2, 2, sample_size)
    X2 = np.random.uniform(-2, 2, sample_size)
    X3 = np.random.uniform(-2, 2, sample_size)
    X4 = np.random.uniform(-2, 2, sample_size)
    X5 = np.random.uniform(-2, 2, sample_size)
    
    # Probabilitiy of participation - affected by X1 and X2
    p_S = (np.exp(beta0 + beta1*X1 + beta2*X2)) / (1 + np.exp(beta0 + beta1*X1 + beta2*X2)) 

    # Study indicator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X1, X2, X3, X4, X5))
    S_trial = data[data[:,0]==1, 0]
    X_trial = np.c_[data[data[:,0]==1,1],
                    data[data[:,0]==1,2],
                    data[data[:,0]==1,3],
                    data[data[:,0]==1,4],
                    data[data[:,0]==1,5]]
    
    # Random treatment     
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE - a function of X1 and X2
    tau_X_trial = 1 + X_trial[:,0] + X_trial[:,1] + X_trial[:,0]**2 + X_trial[:,1]**2

    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + (X_trial[:, 0]**2 + X_trial[:, 1]**2 + X_trial[:,2]**2 + X_trial[:,3]**2 + X_trial[:,4]**2) + epsilon_trial

    ##############################
    ##### Observationla data #####
    ##############################

    # Baseline covariates 
    X1_observational = np.random.uniform(-2, 2, sample_size)
    X2_observational = np.random.uniform(-2, 2, sample_size)
    X3_observational = np.random.uniform(-2, 2, sample_size)
    X4_observational = np.random.uniform(-2, 2, sample_size)
    X5_observational = np.random.uniform(-2, 2, sample_size)
    X_observational = np.c_[X1_observational, X2_observational, X3_observational,
                            X4_observational, X5_observational]
    
    # Study indicator
    S_observational = np.zeros(X1_observational.shape[0])

    # Biased treatment - a function of X1 and X2
    logit_e_X_0 = (X1_observational + X2_observational)
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X1_observational)[0])
    
    # CATE - a function of X1 and X2
    tau_X_observational = 1 + X1_observational + X2_observational + X1_observational**2 + X2_observational**2
    
    # Unobserved confounder  - a function of X1 and X2
    U_observational = (2 * A_observational - 1) * ((X1_observational + X2_observational)) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X1_observational**2 + X2_observational**2 + X3_observational**2 + X4_observational**2 + X5_observational**2 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation


def data_simulation3(sample_size, beta0, beta1, beta2):
    """
    Simulate observational and trial data according to a specified logistic participation model.
    
        Confounding: Complex
        Y(0): Complex 
        Y(1): Complex
        CATE: Simple

    Parameters:
    sample_size : int
        Number of observational samples to generate. The trial sample size will be a fraction of that sample size
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariates (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

    ######################
    ##### Trial data #####
    ######################

    # Baseline covatiates 
    X1 = np.random.uniform(-2, 2, sample_size)
    X2 = np.random.uniform(-2, 2, sample_size)
    X3 = np.random.uniform(-2, 2, sample_size)
    X4 = np.random.uniform(-2, 2, sample_size)
    X5 = np.random.uniform(-2, 2, sample_size)
    
    # Probabilitiy of participation - affected by X1 and X2
    p_S = (np.exp(beta0 + beta1*X1 + beta2*X2)) / (1 + np.exp(beta0 + beta1*X1 + beta2*X2)) 

    # Study indicator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X1, X2, X3, X4, X5))
    S_trial = data[data[:,0]==1, 0]
    X_trial = np.c_[data[data[:,0]==1,1],
                    data[data[:,0]==1,2],
                    data[data[:,0]==1,3],
                    data[data[:,0]==1,4],
                    data[data[:,0]==1,5]]
    
    # Random treatment 
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE - a function of X1 and X2
    tau_X_trial = 1 + X_trial[:,0] + X_trial[:,1]

    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + (X_trial[:, 0]**2 + X_trial[:, 1]**2 + X_trial[:,2]**2 + X_trial[:,3]**2 + X_trial[:,4]**2)  + epsilon_trial

    ##############################
    ##### Observationla data #####
    ##############################

    # Baseline covariates 
    X1_observational = np.random.uniform(-2, 2, sample_size)
    X2_observational = np.random.uniform(-2, 2, sample_size)
    X3_observational = np.random.uniform(-2, 2, sample_size)
    X4_observational = np.random.uniform(-2, 2, sample_size)
    X5_observational = np.random.uniform(-2, 2, sample_size)
    X_observational = np.c_[X1_observational, X2_observational, X3_observational,
                            X4_observational, X5_observational]
    
    # Study indicator
    S_observational = np.zeros(X1_observational.shape[0])

    # Biased treatment - a function of X1 and X2
    logit_e_X_0 = (X1_observational + X2_observational)
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X1_observational)[0])
    
    # Unobserved confounder  - a function of X1 and X2
    tau_X_observational = 1 + X1_observational + X2_observational
    U_observational = (2 * A_observational - 1) *(np.sin(X1_observational) + np.sin(X2_observational)) + norm.rvs(0, 1, size=np.shape(X_observational)[0])

    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X1_observational**2 + X2_observational**2 + X3_observational**2 + X4_observational**2 + X5_observational**2 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation


def data_simulation4(sample_size, beta0, beta1, beta2):
    """
    Simulate observational and trial data according to a specified logistic participation model.
    
        Confounding: Complex
        Y(0): Complex 
        Y(1): Complex
        CATE: Complex

    Parameters:
    sample_size : int
        Number of observational samples to generate. The trial sample size will be a fraction of that sample size
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariates (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

    ######################
    ##### Trial data #####
    ######################

    # Baseline covatiates 
    X1 = np.random.uniform(-2, 2, sample_size)
    X2 = np.random.uniform(-2, 2, sample_size)
    X3 = np.random.uniform(-2, 2, sample_size)
    X4 = np.random.uniform(-2, 2, sample_size)
    X5 = np.random.uniform(-2, 2, sample_size)
    
    # Probabilitiy of participation - affected by X1 and X2
    p_S = (np.exp(beta0 + beta1*X1 + beta2*X2)) / (1 + np.exp(beta0 + beta1*X1 + beta2*X2)) 

    # Study indicator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X1, X2, X3, X4, X5))
    S_trial = data[data[:,0]==1, 0]
    X_trial = np.c_[data[data[:,0]==1,1],
                    data[data[:,0]==1,2],
                    data[data[:,0]==1,3],
                    data[data[:,0]==1,4],
                    data[data[:,0]==1,5]]
    
    # Random treatment   
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE - a function of X1 and X2
    tau_X_trial = 1 + X_trial[:,0] + X_trial[:,1] + X_trial[:,0]**2 + X_trial[:,1]**2

    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + (X_trial[:, 0]**2 + X_trial[:, 1]**2 + X_trial[:,2]**2 + X_trial[:,3]**2 + X_trial[:,4]**2) + epsilon_trial

    ##############################
    ##### Observationla data #####
    ##############################

    # Baseline covariates 
    X1_observational = np.random.uniform(-2, 2, sample_size)
    X2_observational = np.random.uniform(-2, 2, sample_size)
    X3_observational = np.random.uniform(-2, 2, sample_size)
    X4_observational = np.random.uniform(-2, 2, sample_size)
    X5_observational = np.random.uniform(-2, 2, sample_size)
    X_observational = np.c_[X1_observational, X2_observational, X3_observational,
                            X4_observational, X5_observational]
    
    # Study indicator
    S_observational = np.zeros(X1_observational.shape[0])

    # Biased treatment - a function of X1 and X2
    logit_e_X_0 = (X1_observational + X2_observational)
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X1_observational)[0])
    
    # CATE - a function of X1 and X2
    tau_X_observational = 1 + X1_observational + X2_observational + X1_observational**2 + X2_observational**2
    
    # Unobserved confounder  - a function of X1 and X2
    U_observational = (2 * A_observational - 1) *(np.sin(X1_observational) + np.sin(X2_observational)) + norm.rvs(0, 1, size=np.shape(X_observational)[0])

    # Observed outcomes - A function of X1, X2, X3, X4, X5
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X1_observational**2 + X2_observational**2 + X3_observational**2 + X4_observational**2 + X5_observational**2 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation