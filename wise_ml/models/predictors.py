#%%
import os
import pickle
from joblib import dump, load

from tensorflow import keras

#%%
def get_low_redshift_predictor():
    """ 
    Returns
    --------------
    model : tensorflow.python.keras.engine.sequential.Sequential
        Low redshift tensorflow model
    """

    model = keras.models.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lr_predictor'))
    return model


#%%
def get_redshift_predictor():
    """ 
    Returns
    --------------
    model : tensorflow.python.keras.engine.sequential.Sequential
        full redshift tensorflow model
    """

    model = keras.models.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'r_predictor'))
    return model

# %%
def get_r_scaler(size):
    """ 
    Parameters
    --------------
    size : int 
        Either 2 or 4 corresponding to 2 or 4 scaling
    Returns
    --------------
    model : sklearn.preprocessing.StandardScaler
        Scaler used to scale W1W2W3W4 bands before passing into neural network
    """
    if size == 4:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'r_scaler'), 'rb') as f:
            scaler = pickle.load(f)
    elif size == 2:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '2r_scaler'), 'rb') as f:
            scaler = pickle.load(f)
    return scaler
# %%
def get_lr_scaler(size):
    """ 
    Parameters
    --------------
    size : int 
        Either 2 or 4 corresponding to 2 or 4 scaling
    Returns
    --------------
    model : sklearn.preprocessing.StandardScaler
        Scaler used to scale W1W2W3W4 bands before passing into neural network
    """
    if size == 4:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lr_scaler'), 'rb') as f:
            scaler = pickle.load(f)
    elif size == 2:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '2lr_scaler'), 'rb') as f:
            scaler = pickle.load(f)
    return scaler

# %%
def predict_high_redshift(spec_arr):
    """ 
    Parameters
    --------------
    spec_arr : numpy.array
        Nx4 or Nx2 array containing magnitudes of W1, W2, W3, W4 bands or W1, W2 bands
    
    Returns
    --------------
    pred_redshift : numpy.array
        N length array contained predicted array values
    """

    if spec_arr.shape[1] == 4:
        model = get_redshift_predictor()
        scaler = get_r_scaler(4)
        scaled_spec_arr = scaler.transform(spec_arr)
        pred_redshift = model.predict(scaled_spec_arr)
    elif spec_arr.shape[1] == 2:
        model = load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '2r_predictor'))
        scaler = get_r_scaler(2)
        scaled_spec_arr = scaler.transform(spec_arr)
        pred_redshift = model.predict(scaled_spec_arr)
    else:
        raise Exception("Invalid input array shape should be Nx2 or Nx4")

    return pred_redshift

# %%
def predict_low_redshift(spec_arr):
    """ 
    Parameters
    --------------
    spec_arr : numpy.array
        Nx4 or Nx2 array containing magnitudes of W1, W2, W3, W4 bands, or W2 W2 bands
    
    Returns
    --------------
    pred_redshift : numpy.array
        N length array contained predicted array values
    """

    if spec_arr.shape[1] == 4:
        model = get_redshift_predictor()
        scaler = get_lr_scaler(4)
        scaled_spec_arr = scaler.transform(spec_arr)
        pred_redshift = model.predict(scaled_spec_arr)
    elif spec_arr.shape[1] == 2:
        model = load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '2lr_predictor'))
        scaler = get_lr_scaler(2)
        scaled_spec_arr = scaler.transform(spec_arr)
        pred_redshift = model.predict(scaled_spec_arr)
    else:
        raise Exception("Invalid input array shape should be Nx2 or Nx4")

    return pred_redshift

# %%
def predict_redshift(spec_arr):
    """ 
    Parameters
    --------------
    spec_arr : numpy.array
        Nx4 or Nx2 array containing magnitudes of W1, W2, W3, W4 bands or W1, W2 bands
    
    Returns
    --------------
    pred_redshift : numpy.array
        N length array contained predicted array values
    """

    if spec_arr.shape[1] == 4:
        # Predicting higher
        high_redshift = predict_high_redshift(spec_arr)

        # Predicting lower
        low_redshift = predict_low_redshift(spec_arr)

        pred_redshift = (high_redshift + low_redshift) / 2

    elif spec_arr.shape[1] == 2:
        # Predicting higher
        high_redshift = predict_high_redshift(spec_arr)

        # Predicting lower
        low_redshift = predict_low_redshift(spec_arr)

        pred_redshift = (high_redshift + low_redshift) / 2
    
    else:
        raise Exception("Invalid input array shape should be Nx2 or Nx4")

    return pred_redshift
