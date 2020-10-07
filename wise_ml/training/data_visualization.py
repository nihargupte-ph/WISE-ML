# %%
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense


import misc_functions

# %%
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()

# %%
target = train_test_W1W2W3W4['REDSHIFT']
features = train_test_W1W2W3W4.iloc[:, 2:10:2]

# %%
# Plotting redshift vs each of 4 mags
n_samples = 1000
n_divisions = int(target.size/n_samples)
fig, axs = plt.subplots(4, figsize=(8, 6))
fig.tight_layout()
axs[0].set_ylabel("Redshift")
axs[0].set_xlabel("W1")
axs[0].scatter(features.iloc[::n_divisions, 0], target[::n_divisions], s=.1)
axs[1].set_ylabel("Redshift")
axs[1].set_xlabel("W2")
axs[1].scatter(features.iloc[::n_divisions, 1], target[::n_divisions], s=.1)
axs[2].scatter(features.iloc[::n_divisions, 2], target[::n_divisions], s=.1)
axs[2].set_ylabel("Redshift")
axs[2].set_xlabel("W3")
axs[3].scatter(features.iloc[::n_divisions, 3], target[::n_divisions], s=.1)
axs[3].set_ylabel("Redshift")
axs[3].set_xlabel("W4")

# %%
fig, axs = plt.subplots(5, figsize=(8, 6))
fig.tight_layout()
sn.boxplot(x=train_test_W1W2W3W4['REDSHIFT'], ax=axs[0])
sn.boxplot(x=train_test_W1W2W3W4['W1mag'], ax=axs[1])
sn.boxplot(x=train_test_W1W2W3W4['W2mag'], ax=axs[2])
sn.boxplot(x=train_test_W1W2W3W4['W3mag'], ax=axs[3])
sn.boxplot(x=train_test_W1W2W3W4['W4mag'], ax=axs[4])
# %%
fig, axs = plt.subplots(5, figsize=(8, 6))
fig.tight_layout()
axs[0].set_xlabel("W1")
sn.distplot(x=train_test_W1W2W3W4['W1mag'], ax=axs[0])
axs[1].set_xlabel("W2")
sn.distplot(x=train_test_W1W2W3W4['W2mag'], ax=axs[1])
axs[2].set_xlabel("W3")
sn.distplot(x=train_test_W1W2W3W4['W3mag'], ax=axs[2])
axs[3].set_xlabel("W4")
sn.distplot(x=train_test_W1W2W3W4['W4mag'], ax=axs[3])
axs[4].set_xlabel("Redshift")
sn.distplot(x=train_test_W1W2W3W4['REDSHIFT'], ax=axs[4])

# %%
