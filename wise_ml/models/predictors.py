#%%
import os
import pickle

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
def get_r_scaler():
    """ 
    Returns
    --------------
    model : sklearn.preprocessing.StandardScaler
        Scaler used to scale W1W2W3W4 bands before passing into neural network
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'r_scaler'), 'rb') as f:
        scaler = pickle.load(f)
    return scaler
# %%
def get_lr_scaler():
    """ 
    Returns
    --------------
    model : sklearn.preprocessing.StandardScaler
        Scaler used to scale W1W2W3W4 bands before passing into neural network
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lr_scaler'), 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# %%
def predict_high_redshift(spec_arr):
    """ 
    Parameters
    --------------
    spec_arr : numpy.array
        Nx4 array containing magnitudes of W1, W2, W3, W4 bands
    
    Returns
    --------------
    pred_redshift : numpy.array
        N length array contained predicted array values
    """

    model = get_redshift_predictor()
    scaler = get_r_scaler()
    scaled_spec_arr = scaler.transform(spec_arr)
    pred_redshift = model.predict(scaled_spec_arr)
    return pred_redshift

# %%
def predict_low_redshift(spec_arr):
    """ 
    Parameters
    --------------
    spec_arr : numpy.array
        Nx4 array containing magnitudes of W1, W2, W3, W4 bands
    
    Returns
    --------------
    pred_redshift : numpy.array
        N length array contained predicted array values
    """

    model = get_low_redshift_predictor()
    scaler = get_lr_scaler()
    scaled_spec_arr = scaler.transform(spec_arr)
    pred_redshift = model.predict(scaled_spec_arr)
    return pred_redshift

# %%
def predict_redshift(spec_arr):
    """ 
    Parameters
    --------------
    spec_arr : numpy.array
        Nx4 array containing magnitudes of W1, W2, W3, W4 bands
    
    Returns
    --------------
    pred_redshift : numpy.array
        N length array contained predicted array values
    """

    # Predicting higher
    high_redshift = predict_high_redshift(spec_arr)

    # Predicting lower
    low_redshift = predict_low_redshift(spec_arr)

    pred_redshift = (high_redshift + low_redshift) / 2

    return pred_redshift
