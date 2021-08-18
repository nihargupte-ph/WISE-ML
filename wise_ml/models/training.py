import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

from . import misc_functions



def train_standard_architecture(redshift_range=[0, 1.5], n_inputs=4, save=True):
    """ 
    Training a redshift predictor within a certain range

    Parameters
    --------------
    redshift_range : float 
        The range of which to train the network. For example you can specify only train on redshifts between 0 and 2.5 etc. It should give better results
        the smaller range you have up to a certain extent. Namely, when the number of training samples grows too small you will likely see a decline
        in accuracy. I suggest keeping over 50,000 samples.

    n_inputs : int
        Number of inputs you want to the network. This the number of spectroscopic bands. Should be either 2 or 4 as far as I understand.

    save : bool
        If true saves the model to wise_ml/models

    Returns
    --------------
    predict_redshift : python function
        Python wrapper function which will predict redshift. Call it by doing predict_redshift(NxM numpy array) where N is the number of samples 
        to predict and M is the input size (so in this case either 2 or 4)
    """

    # WISE Dataset from Columbia
    predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()

    pd_truth = ((redshift_range[0] < train_test_W1W2W3W4['REDSHIFT']) & (train_test_W1W2W3W4['REDSHIFT'] < redshift_range[1]))
    target = train_test_W1W2W3W4.loc[pd_truth]['REDSHIFT']

    if n_inputs == 2:
        features = train_test_W1W2W3W4.loc[pd_truth].iloc[:, 2:6:2]
    elif n_inputs == 4:
        features = train_test_W1W2W3W4.loc[pd_truth].iloc[:, 2:10:2]
    else:
        raise Exception("Only n_inputs=4 and n_inputs=2 are currently supported")

    # Preprocessing
    X = features
    y = target.ravel()
    indices = np.arange(target.shape[0])
    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, y, indices, test_size=0.05)


    scaler_x = StandardScaler().fit(X_train)

    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)

    # Model Architecture, you can change the architecture here if you like, my guess is for such a small input layer size there won't really 
    # be much difference
    # Building the model
    model = Sequential()
    model.add(Dense(10, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation=tf.math.abs))
    model.summary()

    #Compiling and Training
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae', 'accuracy'])

    # If you want to view the history you can do stuff with it in here
    history = model.fit(X_train, y_train, epochs=30, batch_size=32,  verbose=1, validation_split=0.1)

   
    # Saving the model and scaler
    model.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "keras_models", f'{redshift_range}_{n_inputs}_predictor.h5'))
    joblib.dump(scaler_x, os.path.join(os.path.dirname(os.path.realpath(__file__)), "keras_models", f'{redshift_range}_{n_inputs}_scaler'))

    def predict_redshift(spec_arr):
        scaled_spec_arr = scaler_x.transform(spec_arr)
        pred_redshift = model.predict(scaled_spec_arr)
        return pred_redshift

    return predict_redshift