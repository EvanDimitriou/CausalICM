import numpy as np
import pandas as pd
from scipy.stats import bernoulli, norm

def data_simulation1(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.
    
        Confounding: Simple
        Y(0): Simple
        Y(1): Simple
        CATE: Simple

    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
=======
        Number of observational samples to generate. The trial sample size will be a fraction of that sample size
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """
<<<<<<< HEAD

    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======
    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate 
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of participation 
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 

    # Study indicator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment 
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial

    # Observed outcomes 
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial

    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate 
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment allocation
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])

    # CATE
    tau_X_observational = 1 + X_observational

    # Unobserved confounder
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])

    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])

    # Observed outcomes 
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation


def data_simulation2(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.

        Confounding: Simple
        Y(0): Simple
        Y(1): Complex
        CATE: Complex

    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
=======
        Number of samples to generate. The trial sample size will be a fraction of that sample size.
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

<<<<<<< HEAD
    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial + X_trial**2
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational + X_observational**2
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======
    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of particopation
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 

    # Study indocator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment allocation
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial + X_trial**2

    # Observed outcomes
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial

    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])

    # CATE
    tau_X_observational = 1 + X_observational + X_observational**2

    # Unobserved confounder
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])

    # Observed outcomes
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation

def data_simulation3(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.


        Confounding: Simple
        Y(0): Complex
        Y(1): Complex
        CATE: Simple


    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
    beta0 : float
        Intercept parameter of the logistic participation model.
=======
        Number of samples to generate. The trial sample size will be a fraction of that sample size.
    beta0 : float
        Intercept parameter of the logistic participation model. 
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """
<<<<<<< HEAD
    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======

    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of particopation
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    
    # Study indocator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment allocation
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial

    # Observed outcomes
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    
    # CATE
    tau_X_observational = 1 + X_observational

    # Unobserved confounder
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation

def data_simulation4(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.


        Confounding: Simple
        Y(0): Complex
        Y(1): Complex
        CATE: Complex


    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
=======
        Number of samples to generate. The trial sample size will be a fraction of that sample size.
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

<<<<<<< HEAD
    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial + X_trial**2
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational + X_observational**2
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======
    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of particopation
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    
    # Study indocator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment allocation
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial + X_trial**2

    # Observed outcomes
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    
    # CATE
    tau_X_observational = 1 + X_observational + X_observational**2
    
    # Unobserved confounder
    U_observational = (2 * A_observational - 1) * X_observational + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation

######


def data_simulation5(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.


        Confounding: Complex
        Y(0): Simple
        Y(1): Simple
        CATE: Simple


    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
=======
        Number of samples to generate. The trial sample size will be a fraction of that sample size.
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

<<<<<<< HEAD
    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational
    U_observational = (2 * A_observational - 1) * np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======
    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of particopation
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    
    # Study indocator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment allocation
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial

    # Observed outcomes
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial

    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    
    # CATE
    tau_X_observational = 1 + X_observational
    
    # Unobserved confounder
    U_observational = (2 * A_observational - 1) * np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation


def data_simulation6(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.


        Confounding: Complex
        Y(0): Simple
        Y(1): Complex
        CATE: Complex


    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
=======
        Number of samples to generate. The trial sample size will be a fraction of that sample size.
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

<<<<<<< HEAD
    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial + X_trial**2
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational + X_observational**2
    U_observational = (2 * A_observational - 1) * np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======
    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of particopation
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    
    # Study indocator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment allocation
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial + X_trial**2

    # Observed outcomes
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial + epsilon_trial


    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    
    # CATE
    tau_X_observational = 1 + X_observational + X_observational**2
    
    # Unobserved confounder
    U_observational = (2 * A_observational - 1) * np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation

def data_simulation7(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.


        Confounding: Complex
        Y(0): Complex
        Y(1): Complex
        CATE: Simple


    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
=======
        Number of samples to generate. The trial sample size will be a fraction of that sample size.
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

<<<<<<< HEAD
    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational
    U_observational = (2 * A_observational - 1) * np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======
    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate 
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of participation 
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    
    # Study indicator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment 
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial

    # Observed outcomes
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate 
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment allocation
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    
    # CATE
    tau_X_observational = 1 + X_observational
    
    # Unobserved confounder
    U_observational = (2 * A_observational - 1) * np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes 
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation

def data_simulation8(sample_size, beta0, beta1):
    """
    Simulate observational and trial data according to a specified logistic participation model.\
    

        Confounding: Complex
        Y(0): Complex
        Y(1): Complex
        CATE: Complex


    Parameters:
    sample_size : int
<<<<<<< HEAD
        Number of samples to generate.
=======
        Number of samples to generate. The trial sample size will be a fraction of that sample size
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    beta0 : float
        Intercept parameter of the logistic participation model.
    beta1 : float
        Slope parameter of the logistic participation model.

    Returns:
    data_trial : numpy.ndarray
        Simulated trial data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    data_observation : numpy.ndarray
        Simulated observational data containing columns for participation indicator (S), covariate (X), treatment assignment (A), and outcome (Y).
    """

<<<<<<< HEAD
    X = np.random.uniform(-2, 2, sample_size)
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    S = bernoulli.rvs(p_S)
    data = np.column_stack((S, X))

    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])
    tau_X_trial = 1 + X_trial + X_trial**2
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    X_observational = np.random.uniform(-2, 2, sample_size)
    S_observational = np.zeros(X_observational.shape[0])
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    tau_X_observational = 1 + X_observational + X_observational**2
    U_observational = (2 * A_observational - 1) *np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create DataFrames
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
    
=======
    ######################
    ##### Trial data #####
    ######################

    # Baseline covariate 
    X = np.random.uniform(-2, 2, sample_size)

    # Probability of participation 
    p_S = (np.exp(beta0 + beta1*X)) / (1 + np.exp(beta0 + beta1*X )) 
    
    # Study indicator
    S = bernoulli.rvs(p_S)

    # Keep only the observations where S=1
    data = np.column_stack((S, X))
    S_trial = data[data[:,0]==1, 0]
    X_trial = data[data[:,0]==1,1]

    # Random treatment 
    A_trial = bernoulli.rvs(0.5, size=np.shape(X_trial)[0])

    # CATE
    tau_X_trial = 1 + X_trial + X_trial**2

    # Observed outcomes
    epsilon_trial = norm.rvs(0, 1.0, np.shape(X_trial)[0])
    Y_trial = A_trial * tau_X_trial + X_trial**2 - 1 + epsilon_trial

    ##############################
    ##### Observational data #####
    ##############################

    # Baseline covariate 
    X_observational = np.random.uniform(-2, 2, sample_size)

    # Study indicator
    S_observational = np.zeros(X_observational.shape[0])

    # Biased treatment allocation
    logit_e_X_0 = X_observational
    A_observational = bernoulli.rvs(1 / (1 + np.exp(-logit_e_X_0)), size=np.shape(X_observational)[0])
    
    # CATE
    tau_X_observational = 1 + X_observational + X_observational**2
    
    # Unobserved confounder
    U_observational = (2 * A_observational - 1) *np.sin(X_observational-1) + norm.rvs(0, 1, size=np.shape(X_observational)[0])
    
    # Observed outcomes 
    epsilon_observational = norm.rvs(0, 1, size=np.shape(X_observational)[0])
    Y_observational = A_observational * tau_X_observational + X_observational**2 - 1 + U_observational + epsilon_observational

    # Create Datasets
    data_trial = np.column_stack((S_trial, X_trial, A_trial, Y_trial))
>>>>>>> dbe61ded73cb8aa23fc3626b6b3112b5e9c5cb5f
    data_observation = np.column_stack((S_observational, X_observational, A_observational, Y_observational))

    return data_trial, data_observation