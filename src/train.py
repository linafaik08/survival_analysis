import itertools
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split

def split_train_test(df, cols_x, col_target, test_size=0.3, col_stratify=None, random_state=None):
    Xy = df[cols_x+["censored", col_target]].dropna()

    stratify = None if col_stratify is None else Xy[col_stratify]
    Xy_train, Xy_test = train_test_split(Xy, test_size=test_size, stratify=stratify, random_state=random_state)


    y_train = np.array(list(zip(Xy_train.censored, Xy_train[col_target])),
                       dtype=[('censored', '?'), (col_target, '<f8')])
    y_test = np.array(list(zip(Xy_test.censored, Xy_test[col_target])),
                       dtype=[('censored', '?'), (col_target, '<f8')])

    return Xy_train, Xy_test, y_train, y_test