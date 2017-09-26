# -*- coding: utf-8 -*-
"""
@author: ML-Group-6 --- Girish, Sashank, Ryan, Chengle
"""

from csv import reader
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sklm
import pandas as pd

# Load a CSV file


def load_csv(filename):
    dataset = pd.read_csv(filename, names=['id', 'price',
                                           'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade'])
    return dataset

# Split the dataset into 5 folds


def split_5fold(dataset):
    s = 0.2
    l = len(dataset)
    set_1 = dataset[int(0 * s * l): int(1 * s * l)]
    set_2 = dataset[int(1 * s * l): int(2 * s * l)]
    set_3 = dataset[int(2 * s * l): int(3 * s * l)]
    set_4 = dataset[int(3 * s * l): int(4 * s * l)]
    set_5 = dataset[int(4 * s * l): int(5 * s * l)]
    return [set_1, set_2, set_3, set_4, set_5]

# Global Variables
xs1 = list()
xs2 = list()
xs3 = list()
xs4 = list()
xs5 = list()
ys = list()
yhats = list()
regr = linear_model.LinearRegression()
# Multiple Linear Regression on Dataset

# Load and Prepare Data
filename = 'kc_house_data_pruned_nohead.csv'
dataset = load_csv(filename)


# Create linear regression object
regr = linear_model.LinearRegression()


def run_regr():
    evs_avg, mae_avg, mse_avg, rmse_avg, mdae_avg, rsq_avg = 0, 0, 0, 0, 0, 0
    for i in range(0, 5):
        print('Run : %d' % (i + 1))
        sets = split_5fold(dataset)
        test = sets.pop(i)
        for i in range(0, 5):
            sets = split_5fold(dataset)
            test = sets.pop(i)
            frames = [sets[0], sets[1], sets[2], sets[3]]
    # Evaluate an algorithm using a train/test split
        train = pd.concat(frames)
        kc_house_y_test = [row['price'] for index, row in test.iterrows()]
        test_set = [[row['bedrooms'], row['bathrooms'],
                     row['sqft_living'], row['sqft_lot'], row['grade']] for index, row in test.iterrows()]
        test_set = np.asarray(test_set)
        train_x = [[row['bedrooms'], row['bathrooms'],
                    row['sqft_living'], row['sqft_lot'], row['grade']] for index, row in train.iterrows()]
        train_y = [row['price'] for index, row in train.iterrows()]
        regr.fit(train_x, train_y)
        kc_house_y_pred = regr.predict(test_set)

        global yhats
        yhats = kc_house_y_pred

        # The Coefficients
        print('Coefficients : \n', regr.coef_)
        # Explained Variance Regression Score
        evs = float(sklm.explained_variance_score(
            kc_house_y_test, kc_house_y_pred))
        print('Explained Variance Regression Score : %.3f' % evs)
        # Mean Absolute Error
        mae = float(sklm.mean_absolute_error(kc_house_y_test, kc_house_y_pred))
        print('Mean Absolute Error : %.3f' % mae)
        # Mean Squared Error
        mse = float(sklm.mean_squared_error(kc_house_y_test, kc_house_y_pred))
        print('Mean Squared Error : %.3f' % mse)
        # Root Mean Squared Error
        rmse = float(sqrt(sklm.mean_squared_error(
            kc_house_y_test, kc_house_y_pred)))
        print('Root Mean Squared Error : %.3f' % rmse)
        # Median Absolute Error
        mdae = float(sklm.median_absolute_error(
            kc_house_y_test, kc_house_y_pred))
        print('Median Absolute Error : %.3f' % mdae)
        # R^2 (Coefficient of Determination) Regression Score
        rsq = float(sklm.r2_score(kc_house_y_test, kc_house_y_pred))
        print('R^2 Regression Score : %.3f' % rsq)

        evs_avg += evs
        mae_avg += mae
        mse_avg += mse
        rmse_avg += rmse
        mdae_avg += mdae
        rsq_avg += rsq

        print('\n')

    print('Average Explained Variance Regression Score : %.3f' %
          float(evs_avg / 5))
    print('Average Mean Absolute Error : %.3f' % float(mae_avg / 5))
    print('Average Mean Squared Error : %.3f' % float(mse_avg / 5))
    print('Average Root Mean Squared Error : %.3f' % float(rmse_avg / 5))
    print('Average Median Absolute Error : %.3f' % float(mdae_avg / 5))
    print('Average R^2 Regression Score : %.3f' % float(rsq_avg / 5))
    print('\n')

    return None

# Evaluate Algorithm
print('Price vs. Bedrooms, Bathrooms, Living Sqft, Lot SqFt, Grade')
run_regr()
