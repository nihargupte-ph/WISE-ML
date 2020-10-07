import os
import pickle

import pandas as pd

def load_data():
    """ 
    Returns
    -------------
    predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 : pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, 
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "predict_W1W2.pkl"), "rb") as f:
        predict_W1W2 = pickle.load(f)

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "predict_W1W2W3W4.pkl"), "rb") as f:
        predict_W1W2W3W4 = pickle.load(f)

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "train_test_W1W2.pkl"), "rb") as f:
        train_test_W1W2 = pickle.load(f)

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "train_test_W1W2W3W4.pkl"), "rb") as f:
        train_test_W1W2W3W4 = pickle.load(f)

    return predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4
