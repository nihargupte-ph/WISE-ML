#%%
import os
import pickle
from joblib import dump, load
import joblib

from tensorflow import keras
import tensorflow as tf

#%%
def get_redshift_predictor(redshift_range=[0, 1.5], n_inputs=4):
    """ 
    For getting an already saved redshift predictor. 

    Parameters
    --------------
    redshift_range : list 
        Loads redshift predictor within the specified range. Default is [0, 1.5]

    n_inputs : int
        Number of inputs you want to the network. This the number of spectroscopic bands. Should be either 2 or 4 as far as I understand.

    Returns
    --------------
    model : tensorflow.python.keras.engine.sequential.Sequential
        full redshift tensorflow model
    """

    model = keras.models.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), "keras_models", f'{redshift_range}_{n_inputs}_predictor.h5'),
    custom_objects={"abs":tf.math.abs})
    return model

# %%
def get_r_scaler(redshift_range=[0, 1.5], n_inputs=4):
    """ 
    Parameters
    --------------
    redshift_range : list 
        Loads redshift predictor within the specified range. Default is [0, 1.5]

    n_inputs : int
        Number of inputs you want to the network. This the number of spectroscopic bands. Should be either 2 or 4 as far as I understand.

    Returns
    --------------
    model : sklearn.preprocessing.StandardScaler
        Scaler used to scale W1W2W3W4 bands before passing into neural network
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "keras_models", f'{redshift_range}_{n_inputs}_scaler'), 'rb') as f:
        scaler = joblib.load(f)

    return scaler


# %%
def predict_redshift(spec_arr, redshift_range=[0, 1.5], n_inputs=4):
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

    scaler = get_r_scaler(redshift_range=redshift_range, n_inputs=n_inputs)
    model = get_redshift_predictor(redshift_range=redshift_range, n_inputs=n_inputs)

    scaled_spec_arr = scaler.transform(spec_arr)
    pred_redshift = model.predict(scaled_spec_arr)
    
    return pred_redshift
