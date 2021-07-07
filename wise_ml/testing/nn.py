# %%
import pickle
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
import astropy.io
import astropy.coordinates
import astropy.units as u
from keras.layers import Dense, Input, Dropout
import keras.models 
import keras.layers
from keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

import misc_functions

# %%
def create_dropout_predict_function(model, dropout):
    """
    modified from: 
    https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf

    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers
    
    Returns
    predict_with_dropout : keras function for predicting with dropout
    """
    
    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"]=="Dropout":
            layer["config"]["rate"] = dropout


    # Create a new model with specified dropout
    if isinstance(model, keras.models.Sequential):
        # Sequential
        model_dropout = keras.models.Sequential.from_config(conf)
    else:
        # Functional
        model_dropout = keras.models.Model.from_config(conf)
    model_dropout.set_weights(model.get_weights()) 
    
    # Create a function to predict with the dropout on
    def predict_with_dropout(inputs, model_dropout=model_dropout):
        K.set_learning_phase(1)
        inputs = K.constant(inputs)
        outputs = model_dropout.call(inputs, training=False).numpy()
        return outputs

    return predict_with_dropout

def pipeline_dropout_predictions(X, dropout=0.5, num_iter=20, pipeline=None, scaler=None, model=None):

    if scaler == None and model == None and pipeline == None:
        raise Exception("scaler/model and pipeline can't all be Nonetype")

    if scaler == None or model == None:
        scaler = pipeline.named_steps['preprocess']
        model = pipeline.named_steps['ann'].model

    num_samples = X.shape[0]
    predictions = np.zeros((num_samples, num_iter))

    predict_with_dropout = create_dropout_predict_function(model, dropout)

    for i in range(num_iter):
        inputs = scaler.transform(X)
        pred = predict_with_dropout(inputs).reshape(-1)
        predictions[:,i] = pred

    return predictions

def plot_confidence(dropout_predictions, actual_predictions, target, **kwargs):
    median =  np.repeat(np.median(dropout_predictions, 1)[:, np.newaxis], dropout_predictions.shape[1], axis=1)
    dropout_distribution = dropout_predictions - median
    error_distribution = actual_predictions.ravel() - target.ravel()
    dropout_distribution = dropout_distribution.ravel()
    error_distribution = error_distribution.ravel()
    plt.ylabel('Relative Frequency')
    plt.xlabel('Error')
    plt.hist(error_distribution, weights=np.ones_like(error_distribution) / error_distribution.size, alpha=.5, label='actual error', **kwargs)
    plt.hist(dropout_distribution, weights=np.ones_like(dropout_distribution) / dropout_distribution.size, alpha=.5, label='dropout prediction - median dropout prediction', **kwargs)
    plt.legend()

def get_confidence(dropout_predictions, ci=0.95):
    lower_lim = np.quantile(dropout_predictions, 0.5-ci/2, axis=1)
    upper_lim = np.quantile(dropout_predictions, 0.5+ci/2, axis=1)
    return lower_lim, upper_lim


# %%
# WISE Dataset from Columbia
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()
# error = train_test_W1W2W3W4[['e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag']]
# total_error = np.sum(error.to_numpy(), axis=1)
# mask = total_error < 1

# target = train_test_W1W2W3W4['REDSHIFT'][mask]
# features = train_test_W1W2W3W4.iloc[:, 2:10:2][mask]

target = train_test_W1W2W3W4.loc[train_test_W1W2W3W4['REDSHIFT'] < 1.5]['REDSHIFT']
features = train_test_W1W2W3W4.loc[train_test_W1W2W3W4['REDSHIFT'] < 1.5].iloc[:, 2:10:2]


#error = train_test_W1W2W3W4.iloc[:, 3:11:2]
#sample_weight = 1 / (error.iloc[:, 0] + error.iloc[:, 1] + .35*error.iloc[:, 2] + .1*error.iloc[:, 3]).to_numpy()

# Preprocessing
X = features
y = target.ravel()
indices = np.arange(target.shape[0])
X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, y, indices, test_size=0.05)
# sample_weight = sample_weight[ind_train].reshape(-1, 1)

# scaler_sample = MinMaxScaler().fit(sample_weight)
scaler_x = StandardScaler().fit(X_train)

# sample_weight = scaler_sample.transform(sample_weight)
# sample_weight = sample_weight.reshape(-1)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

# %%
#Building the model
model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#Compiling and Training
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae', 'accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=32,  verbose=1, validation_split=0.1)


# %%
def create_dropout_predict_function(model, dropout):
    """
    modified from: 
    https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf

    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers
    
    Returns
    predict_with_dropout : keras function for predicting with dropout
    """
    
    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"]=="Dropout":
            layer["config"]["rate"] = dropout


    # Create a new model with specified dropout
    if isinstance(model, keras.models.Sequential):
        # Sequential
        model_dropout = keras.models.Sequential.from_config(conf)
    else:
        # Functional
        model_dropout = keras.models.Model.from_config(conf)
    model_dropout.set_weights(model.get_weights()) 
    
    # Create a function to predict with the dropout on
    def predict_with_dropout(inputs, model_dropout=model_dropout):
        K.set_learning_phase(1)
        inputs = K.constant(inputs)
        outputs = model_dropout.call(inputs, training=False).numpy()
        return outputs

    return predict_with_dropout

def pipeline_dropout_predictions(X, dropout=0.5, num_iter=20, pipeline=None, scaler=None, model=None):

    if scaler == None and model == None and pipeline == None:
        raise Exception("scaler/model and pipeline can't all be Nonetype")

    if scaler == None or model == None:
        scaler = pipeline.named_steps['preprocess']
        model = pipeline.named_steps['ann'].model

    num_samples = X.shape[0]
    predictions = np.zeros((num_samples, num_iter))

    predict_with_dropout = create_dropout_predict_function(model, dropout)

    for i in range(num_iter):
        inputs = scaler.transform(X)
        pred = predict_with_dropout(inputs).reshape(-1)
        predictions[:,i] = pred

    return predictions

def plot_confidence(dropout_predictions, actual_predictions, target, **kwargs):
    median =  np.repeat(np.median(dropout_predictions, 1)[:, np.newaxis], dropout_predictions.shape[1], axis=1)
    dropout_distribution = dropout_predictions - median
    error_distribution = actual_predictions.ravel() - target.ravel()
    dropout_distribution = dropout_distribution.ravel()
    error_distribution = error_distribution.ravel()
    plt.ylabel('Relative Frequency')
    plt.xlabel('Error')
    plt.hist(error_distribution, weights=np.ones_like(error_distribution) / error_distribution.size, alpha=.5, label='actual error', **kwargs)
    plt.hist(dropout_distribution, weights=np.ones_like(dropout_distribution) / dropout_distribution.size, alpha=.5, label='dropout prediction - median dropout prediction', **kwargs)
    plt.legend()

def get_confidence(dropout_predictions, ci=0.95):
    lower_lim = np.quantile(dropout_predictions, 0.5-ci/2, axis=1)
    upper_lim = np.quantile(dropout_predictions, 0.5+ci/2, axis=1)
    return lower_lim, upper_lim

X = features
y = target.ravel()
indices = np.arange(target.shape[0])
# These are not transformed just make sure by printing them though
X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, y, indices, test_size=0.05)

dropout_predictions = pipeline_dropout_predictions(X_test, dropout=0.5, scaler=scaler_x, model=model)
actual_predictions = model.predict(X_test).reshape(-1)
plot_confidence(dropout_predictions, actual_predictions, y_test, bins=25)


# %%
# prediction
y_pred_without_dropout = model_without_dropout.predict(x_test)
y_pred_with_dropout = model_with_dropout.predict(x_test)

# plotting
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(x_train, y_train, s=10, label='train data')
ax.plot(x_test, x_test, ls='--', label='test data', color='green')
ax.plot(x_test, y_pred_without_dropout, label='predicted ANN - R2 {:.2f}'.format(r2_score(x_test, y_pred_without_dropout)), color='red')
ax.plot(x_test, y_pred_with_dropout, label='predicted ANN Dropout - R2 {:.2f}'.format(r2_score(x_test, y_pred_with_dropout)), color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('test data')


# %%
print(history.history.keys())
y_pred = model.predict(X_test)
print(r2_score(y_pred, y_test))
mean_error = np.nanmean(np.abs(y_pred[::10] - y_test[::10]))
print(mean_error)

# MAE
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mean absolute error')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plotting linear regression
plt.scatter(y_pred, y_test, s=1)
plt.plot([np.min(y_pred), np.max(y_pred)], [np.min(y_pred), np.max(y_pred)], zorder=5, color='orange')

# %%
model.save()

#%%
x, y = y_reg, y_trgInput, Dropout
#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.') 
plt.title("WISE ML z < 2.5")
plt.ylabel("Actual Redshift")
plt.xlabel("Predicted Redshift")
plt.plot([0, 2.5], [0, 2.5])
plt.savefig("/home/n/Documents/Research/WISE-ML/plots/WISEML_Density_Scatter_smallz.png")
plt.show()

# %%
x, y = y_test, model.predict(X_test).reshape(-1)
#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.') 
plt.title("ANNZ2 results")
plt.ylabel("Actual Redshift")
plt.xlabel("Predicted Redshift")
plt.plot([0, 1], [0, 1], c='r')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("/home/n/Documents/Research/WISE-ML/plots/WISEML_Density_Scatter_regular.png")
plt.show()