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
plt.savefig("/home/n/Documents/Research/WISE-ML/plots/ANNZ2_redshift.pdf")
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
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
plot_df = train_test_W1W2W3W4.loc[train_test_W1W2W3W4['REDSHIFT'] < 2]
projected = pca.fit_transform(plot_df[['W1mag', 'W2mag', 'W3mag', 'W4mag']])
print(pca.explained_variance_ratio_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projected[:, 0], projected[:, 1], plot_df['REDSHIFT'], c=plot_df['REDSHIFT'], s=2)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('Redshift')
plt.show()
# %%
plot_df = train_test_W1W2.loc[train_test_W1W2['REDSHIFT'] < 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plot_df['W1mag'], plot_df['W2mag'], plot_df['REDSHIFT'], c=plot_df['REDSHIFT'], s=2)
ax.set_xlabel('W1 magnitude')
ax.set_ylabel('W2 magnitude')
ax.set_zlabel('Redshift')

plt.show()

# %%
