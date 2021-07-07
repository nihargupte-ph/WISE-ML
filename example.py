#%%
import numpy as np
import matplotlib.pyplot as plt

from wise_ml.models import predictors
from wise_ml.models import training
from wise_ml.models import misc_functions

# %%
# So this is the old method that you were using earlier. Only difference is I've done away with the low+high redshift averaging now that we have a 
# better estimate on the range of redshift we are expecting. That is if you are only interested in the redshift region between 0 and 1 it is better to
# just use the previously called "low redshift predictor". Now it is called just redshift predictor. To make it more clear you can specify the range
# of redshifts that you want to predict. Right now I've trained a range of [0, 1.5] for both 2 and 4 bands. So if you just want to use the predictor,
# just specify redshift_range = [0, 1.5] and n_inputs = 2 or 4 to predictors.predict_redshift
# Example below
redshift_range = [0, 1.5]
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()
pd_truth = ((redshift_range[0] < train_test_W1W2W3W4['REDSHIFT']) & (train_test_W1W2W3W4['REDSHIFT'] < redshift_range[1]))
y_test = train_test_W1W2W3W4.loc[pd_truth]['REDSHIFT']

# Selecting only bands 1 and 2 since above we had n_inputs = 2. If you want to predict 4 change this 6 
# to a 10 basically
features = train_test_W1W2W3W4.loc[pd_truth].iloc[:, 2:6:2]

# Prediction
y_pred = predictors.predict_redshift(features, redshift_range = redshift_range, n_inputs = 2)
plt.scatter(y_pred, y_test, s=1)
plt.show()

# %%
# This is the training module I just implemented. Basically you can train whatever range you would like while also selecting the number of inputs
# It will save the models as well so later on you can just call it with the scheme described above. As an example try running the following code
# And it will save a new prediction model as well as create a new function called "predict_redshift" here which you can use to predict whatever 
# You want in your juypter notebook or dev environment. 

# Take care naming the functions you can use a __name__ method but make sure the variable scope is well defined I'll leave it to you how you want 
# to do it. You can open up this function and edit things around if you want to play with the architecture, epochs, etc. if you want but if not
# i left it like this to be easy to use

# So just specify the range you are interested in and it will train for that range
redshift_range = [0, 1.5]
# Below select the number of inputs (n_inputs)
predict_redshift = training.train_standard_architecture(redshift_range, n_inputs=2)

# Usage, note it returns a function which you can then call eg
# first getting the data from the dataset and only selecting redshifts within 0 to 1.5
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()
pd_truth = ((redshift_range[0] < train_test_W1W2W3W4['REDSHIFT']) & (train_test_W1W2W3W4['REDSHIFT'] < redshift_range[1]))
y_test = train_test_W1W2W3W4.loc[pd_truth]['REDSHIFT']

# Selecting only bands 1 and 2 since above we had n_inputs = 2. If you want to train 4 change this 6 
# to a 10 basically
features = train_test_W1W2W3W4.loc[pd_truth].iloc[:, 2:6:2]

# Predicting data
y_pred = predict_redshift(features)

# I believe this is the plot Prof Marka suggested. I'm not sure what fonts etc to use but they can be edited here. This is done on the training set
plt.xlabel('Target Redshift')
plt.ylabel('Regressed Redshift')
plt.scatter(y_test, y_pred, s=1)
plt.plot([np.min(y_pred), np.max(y_pred)], [np.min(y_pred), np.max(y_pred)], zorder=5, color='orange')
plt.show()

# %%
# Perhaps more clear is a variant of the above plot but this time encoding density also (density scatter plot)
x, y = y_pred.ravel(), np.array(y_test)
#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
plt.xlabel("Predicted Redshift")
plt.ylabel("Actual Redshift")
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.') 
plt.plot([np.min(y_pred), np.max(y_pred)], [np.min(y_pred), np.max(y_pred)], zorder=5, color='orange')
plt.show()
# %%
# Tried to recreate the histogram you mentioned with the below 0 values and was unable
# Perhaps it was some old model detail? I'm not entirely sure but anyway this one at least
# Gives me positive results
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()
pd_truth = ((redshift_range[0] < train_test_W1W2W3W4['REDSHIFT']) & (train_test_W1W2W3W4['REDSHIFT'] < redshift_range[1]))

train_test_features = train_test_W1W2W3W4.loc[pd_truth].iloc[:, 2:10:2]
predict_features = predict_W1W2W3W4.iloc[:, 2:10:2]

# Predicting data
y_prediction = predictors.predict_redshift(predict_features, redshift_range = redshift_range, n_inputs = 4)
y_train_test = predictors.predict_redshift(train_test_features, redshift_range = redshift_range, n_inputs = 4)

fig, axs = plt.subplots(2)
axs[0].set_xlabel("Regressed Value (Prediction Set)")
axs[0].set_ylabel("Counts")
axs[0].hist(y_prediction, bins=100)
print(np.where(y_pred < 0)) # Empty array

axs[1].set_xlabel("Regressed Value (Training Set)")
axs[1].set_ylabel("Counts")
axs[1].hist(y_train_test, bins=100)
print(np.where(y_test < 0)) # Empty array


# %%
# Notes on NN confidence intervals. As far as I understand we won't be able to generate confidence intervals for such a small network
# This is because when thinning the dropout layers by thinning even one there is such little aleatoric uncertainty in the output that every 
# Thinned network returns the same value. If you are interested you can check out the code in testing/nn.py which is essentially the same code 
# I used in another project to generate the confidence intervals. At the moment it does what I mentioned above. If this doesn't make sense
# no need to worry, the rest of the network will work as intended. 
