# %%
import pickle
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense

import misc_functions


# %%
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()

# target = train_test_W1W2W3W4['REDSHIFT']
# features = train_test_W1W2W3W4.iloc[:, 2:10:2]
target = train_test_W1W2W3W4.loc[train_test_W1W2W3W4['REDSHIFT'] < 2.5]['REDSHIFT']
features = train_test_W1W2W3W4.loc[train_test_W1W2W3W4['REDSHIFT'] < 2.5].iloc[:, 2:6:2]
# target = train_test_W1W2W3W4['REDSHIFT']
# features = train_test_W1W2W3W4.iloc[:, 2:6:2]


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
# Saving the scaler so we can use when predicting
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", 'models', '2lr_scaler'), 'wb') as f:
    pickle.dump(scaler_x, f)

# %%
#Building the model
model = Sequential()
model.add(Dense(4, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#Compiling and Training
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae', 'accuracy'])
history = model.fit(X_train, y_train, epochs=60, batch_size=32,  verbose=1, validation_split=0.1) # , sample_weight=sample_weight)


#%%
df = pd.read_csv("/home/n/Documents/Research/etc_codes/ANNZ/data/photoZ/train/WISE_test_4.csv", header=0, names=['REDSHIFT', 'W1mag', 'e_W1mag', 'W2mag', 'e_W2mag', 'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag'])
y_trg = df["REDSHIFT"]
X_test_ANNZ = df.iloc[:, 1::2]
X_test_ANNZ_scl = scaler_x.transform(X_test_ANNZ)
y_reg = model.predict(X_test_ANNZ_scl).ravel()
err = (y_reg - y_trg) / y_trg
plt.figure(figsize=(8,6))
plt.ylabel("Counts")
plt.xlabel(r"$(z_{reg} - z_{trg})/z_{trg}$")
plt.hist(err, bins=1000, range=[-1, 10])
plt.savefig("/home/n/Documents/Research/WISE-ML/plots/err_hist_ours_1z.pdf")
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
fig, ax = plt.subplots()
df = pd.DataFrame(data={'predicted':y_pred.reshape(-1), 'actual':y_test})
sn.set(color_codes=True)
ax=sn.regplot(x='actual',y='predicted',data=df,  ax=ax, scatter_kws={"s": .5, 'alpha':.5})
plt.show()
