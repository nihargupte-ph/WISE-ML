#%%
import numpy as np

from wise_ml.models import predictors
from wise_ml.training import misc_functions

# Loading data
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()

# Predict redshift using a high predictor 
redshift = predictors.predict_high_redshift(train_test_W1W2W3W4.iloc[:1000, 2:10:2])
# %%
# Creating a 2xN array to compare actual and predicted values
actual_pred_redshift = np.c_[train_test_W1W2W3W4.iloc[:1000, :]['REDSHIFT'], redshift]
print(actual_pred_redshift[100:200:20, :])

# Mean error
print(np.mean(actual_pred_redshift[:, 0] - actual_pred_redshift[:, 1]))
# %%
# Predict redshift using a low predictor
redshift = predictors.predict_low_redshift(train_test_W1W2W3W4.iloc[::10, 2:10:2])

actual_pred_redshift = np.c_[train_test_W1W2W3W4.iloc[::10, :]['REDSHIFT'], redshift]
print(actual_pred_redshift[100:200:20, :])

# Mean error
print(np.mean(actual_pred_redshift[:, 0] - actual_pred_redshift[:, 1]))

# %%
# Predicting using standard predictor
redshift = predictors.predict_redshift(train_test_W1W2W3W4.iloc[::10, 2:10:2])

actual_pred_redshift = np.c_[train_test_W1W2W3W4.iloc[::10, :]['REDSHIFT'], redshift]
print(actual_pred_redshift[:100, :])

# Mean error
print(np.mean(actual_pred_redshift[:, 0] - actual_pred_redshift[:, 1]))
