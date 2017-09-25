# -*- coding: utf-8 -*-
"""
@author: ML-Group-6 --- Girish, Sashank, Ryan, Chengle
"""

# id,bedrooms,bathrooms,sqft_living15,sqft_lot15,grade,price

from random import seed
# from random import randrange
from math import sqrt
import numpy as np
import pandas as pd
# Load a CSV file


def load_csv(filename):
    dataset = pd.read_csv(filename, names=['id', 'price',
                                           'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade'])
    dataset['price'] = dataset['price'].apply(np.log)
    dataset['sqft_lot'] = dataset['sqft_lot'].apply(np.log)
    dataset['sqft_living'] = dataset['sqft_living'].apply(np.log)
    return dataset


# Split the dataset into 5 folds

def split_5fold(dataset):
    s = 0.2
    l = len(dataset)
    set_1 = dataset[int(0 * s * l):int(1 * s * l)]
    set_2 = dataset[int(1 * s * l):int(2 * s * l)]
    set_3 = dataset[int(2 * s * l):int(3 * s * l)]
    set_4 = dataset[int(3 * s * l):int(4 * s * l)]
    set_5 = dataset[int(4 * s * l):int(5 * s * l)]
    return [set_1, set_2, set_3, set_4, set_5]

# Global Variables
ymean = 0

# Calculate Sum of Squared Errors Total


def tsse(values, mean):
    mean = [mean] * len(values)
    sst = np.subtract(values, mean).reshape((len(values), 1))
    sst = np.square(sst)
    sst = sum(sst)
    return sst

# Calculate Sum of Squared Errors Residual


def rsse(values, hats):
    ssr = np.subtract(values, hats).reshape((len(values), 1))
    ssr = np.square(ssr)
    ssr = sum(ssr)
    return ssr

# Calculate Sum of Squared Errors Explained


def esse(hats, mean):
    mean = [mean] * len(hats)
    sse = np.subtract(hats, mean).reshape((len(hats), 1))
    sse = np.square(sse)
    sse = sum(sse)
    return sse

# Calculate R Squared Value


def r_sq(r, t):
    rsq = 1 - (r / t)
    return rsq

# Calculate root mean squared error


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return np.sqrt(mean_error)


# Evaluate an algorithm using a train/test split

def evaluate_algorithm(dataset, algorithm, *args):
    # train, test = train_test_split(dataset, split)
    rmse_avg, sst_avg, ssr_avg, sse_avg, r_sq_avg = 0, 0, 0, 0, 0
    for i in range(0, 5):
        sets = split_5fold(dataset)
        test = sets.pop(i)
        frames = [sets[0], sets[1], sets[2], sets[3]]
# Evaluate an algorithm using a train/test split
        train = pd.concat(frames)
        ytest = [row['price'] for index, row in test.iterrows()]
        test_set = [[row['bedrooms'], row['bathrooms'],
                     row['sqft_living'], row['sqft_lot'], row['grade']] for index, row in test.iterrows()]
        predicted = algorithm(train, test_set, *args)
        rmse = rmse_metric(ytest, predicted)
        rmse_avg += rmse
        sst_avg += tsse(ytest, ymean)
        ssr_avg += rsse(ytest, predicted)
        sse_avg += esse(predicted, ymean)
        r_sq_avg += r_sq(rsse(ytest, predicted), tsse(ytest, ymean))
    return [rmse_avg / 5, sst_avg / 5, ssr_avg / 5, sse_avg / 5, r_sq_avg / 5]

# Calculate the mean value of a list of numbers


def mean(values):
    return np.divide(np.sum(values, axis=0), float(len(values)))

# Calculate the variance of a list of numbers


def variance(values, mean):
    return np.sum(np.square(values - mean))

# Calculate covariance between x and y


def covariance(x, mean_x, y, mean_y):
    covar = np.sum((x - mean_x)
                   * (y - mean_y))
    return covar

# id,bedrooms,bathrooms,sqft_living15,sqft_lot15,grade,price
# Calculate coefficients


def coefficients(dataset):
    x = [[row['bedrooms'], row['bathrooms'],
          row['sqft_living'], row['sqft_lot'], row['grade']] for index, row in dataset.iterrows()]
    y = [row['price'] for index, row in dataset.iterrows()]
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = mean(x)
    ymean = mean(y)
    x = np.swapaxes(x, 0, 1)
    coeffs = [[0 for i in range(len(x))] for j in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                coeffs[i][j] = variance(x[i], x_mean[i])
            elif i > j:
                coeffs[i][j] = coeffs[j][i]
            else:
                coeffs[i][j] = covariance(x[i], x_mean[i], x[j], x_mean[j])
    eq_ycovs = np.array([covariance(y, ymean, x[0], x_mean[0]), covariance(y, ymean, x[1], x_mean[1]), covariance(
        y, ymean, x[2], x_mean[2]), covariance(y, ymean, x[3], x_mean[3]), covariance(y, ymean, x[4], x_mean[4])])
    b_coeffs = np.linalg.solve(coeffs, eq_ycovs)
    b0 = ymean - (np.sum(b_coeffs * x_mean))
    return [b0, b_coeffs]

# Calculate coefficients
# def coefficients(dataset):
#	x = [row[0] for row in dataset]
#	y = [row[1] for row in dataset]
#	x_mean, y_mean = mean(x), mean(y)
#	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
#	b0 = y_mean - b1 * x_mean
#	return [b0, b1]

# Simple linear regression algorithm
# def simple_linear_regression(train, test):
#	predictions = list()
#	b0, b1 = coefficients(train)
#	for row in test:
#		yhat = b0 + b1 * row[0]
#		predictions.append(yhat)
#	return predictions

# Multiple linear regression algorithm


def multiple_linear_regression(train, test):
    predictions = list()
    test = np.asarray(test)
    b0, b_coeffs = coefficients(train)
    for row in test:
        yhat = b0 + np.sum(b_coeffs * row)
        predictions.append(yhat)
    return predictions

# Simple linear regression on dataset
seed(1)
# load and prepare data
filename = 'kc_house_data_pruned_nohead.csv'
dataset = load_csv(filename)
# evaluate algorithm
# rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
results = evaluate_algorithm(dataset, multiple_linear_regression)
print('RMSE: %.3f TSSE: %.3f RSSE: %.3f ESSE: %.3f R^2: %.3f' %
      (results[0], results[1], results[2], results[3], results[4]))
