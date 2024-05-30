import numpy as np
def weighted_mean_squared_error(weight, t, y_true, y_pred0, y_pred1):
    """
    Mean Squared Error for the weighted RCT individuals (necessary for finding the optimal value of rho)

    - weight (numpy.ndarray): The weightes for the RCT paricipants
    - t (int): Treatment value
    - y_true (numpy.ndarray): the observed outcome
    - y_pred0 (numpy.ndarray): The predicted outcome for control individuals
    - y_pred1 (numpy.ndarray): The predicted outcome for treated individuals 
    """
    # Ensure inputs are numpy arrays
    y_true = np.hstack(y_true)
    y_pred0 = np.hstack(y_pred0)
    y_pred1 = np.hstack(y_pred1)
    weight = np.hstack(weight)
    weighted_squared_diff = np.zeros(y_true.shape[0])
    weighted_squared_diff[t==0] = weight[t==0]*((y_true[t==0] - y_pred0[t==0])**2) 
    weighted_squared_diff[t==1] = weight[t==1]*((y_true[t==1] - y_pred1[t==1])**2)  
    wmse = np.mean(weighted_squared_diff)
        
    return wmse
