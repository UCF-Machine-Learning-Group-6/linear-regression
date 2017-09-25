# -*- coding: utf-8 -*-
"""
@author: ML-Group-6 --- Girish, Sashank, Ryan, Chengle
"""

from csv import reader
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Split the dataset into 5 folds
def split_5fold(dataset):
    s = 0.2
    l = len(dataset)
    set_1 = dataset[int(0 * s * l) : int(1 * s * l)]
    set_2 = dataset[int(1 * s * l) : int(2 * s * l)]
    set_3 = dataset[int(2 * s * l) : int(3 * s * l)]
    set_4 = dataset[int(3 * s * l) : int(4 * s * l)]
    set_5 = dataset[int(4 * s * l) : int(5 * s * l)]
    return [set_1, set_2, set_3, set_4, set_5]

# Global Variables
xs = list()
ys = list()
yhats = list()
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
def r_sq(r,t):
    rsq = 1 - (r/t)
    return rsq

# Calculate Root Mean Squared Error
def rmse(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, *args):
    rmse_avg, sst_avg, ssr_avg, sse_avg, r_sq_avg = 0, 0, 0, 0, 0
    for i in range(0,5):
        sets = split_5fold(dataset)
        test = sets.pop(i)
        ytest = [row[6] for row in test]
        global ys
        ys = ytest
        xtest = [row[r] for row in test]
        global xs
        xs = xtest
        train = sets[0] + sets[1] + sets[2] + sets[3]
        test_set = list()
        for row in test:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train, test_set, *args)
        actual = [row[-1] for row in test]
        rms = rmse(actual, predicted)
        rmse_avg += rms
        sst_avg += tsse(ys, ymean)
        ssr_avg += rsse(ys, yhats)
        sse_avg += esse(yhats, ymean)
        r_sq_avg += r_sq(rsse(ys, yhats), tsse(ys, ymean))
    return [rmse_avg/5, sst_avg/5, ssr_avg/5, sse_avg/5, r_sq_avg/5]

# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))
 
# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# dataset - id, bedrooms, bathrooms, sqft_living15, sqft_lot15, grade, price
# Calculate Coefficients
def simple_coefficients(dataset):
    x = [row[r] for row in dataset]
    y = [row[6] for row in dataset]
    x_mean = mean(x)
    y_mean = mean(y)
    global ymean
    ymean = y_mean
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

# Simple Linear Regression Algorithm
def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = simple_coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[r]
        predictions.append(yhat)
    global yhats
    yhats = predictions
    return predictions


# Simple Linear Regression on Dataset

# Load and Prepare Data
filename = 'house_data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
    
# Evaluate Algorithm
for r in range(1,6):
    if r == 1:
        print('Price vs. Bedrooms')
        res_slr = evaluate_algorithm(dataset, simple_linear_regression)
        print('RMSE: %.3f TSSE: %.3f RSSE: %.3f ESSE: %.3f R^2: %.3f \n' % (res_slr[0], res_slr[1], res_slr[2], res_slr[3], res_slr[4]))
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('No. of Bedrooms')
        plt.ylabel('Prices')
        plt.title('Price vs. Bedrooms')
        plt.grid(None, 'both', 'both')
    elif r == 2:
        print('Price vs. Bathrooms')
        res_slr = evaluate_algorithm(dataset, simple_linear_regression)
        print('RMSE: %.3f TSSE: %.3f RSSE: %.3f ESSE: %.3f R^2: %.3f \n' % (res_slr[0], res_slr[1], res_slr[2], res_slr[3], res_slr[4]))
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('No. of Bathrooms')
        plt.ylabel('Prices')
        plt.title('Price vs. Bathrooms')
        plt.grid(None, 'both', 'both')
    elif r == 3:
        print('Price vs. Living SqFt')
        res_slr = evaluate_algorithm(dataset, simple_linear_regression)
        print('RMSE: %.3f TSSE: %.3f RSSE: %.3f ESSE: %.3f R^2: %.3f \n' % (res_slr[0], res_slr[1], res_slr[2], res_slr[3], res_slr[4]))
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('Living SqFt')
        plt.ylabel('Prices')
        plt.title('Price vs. Living SqFt')
        plt.grid(None, 'both', 'both')
    elif r == 4:
        print('Price vs. Lot SqFt')
        res_slr = evaluate_algorithm(dataset, simple_linear_regression)
        print('RMSE: %.3f TSSE: %.3f RSSE: %.3f ESSE: %.3f R^2: %.3f \n' % (res_slr[0], res_slr[1], res_slr[2], res_slr[3], res_slr[4]))
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('Lot SqFt')
        plt.ylabel('Prices')
        plt.title('Price vs. Lot SqFt')
        plt.grid(None, 'both', 'both')
    elif r == 5:
        print('Price vs. Grade')
        res_slr = evaluate_algorithm(dataset, simple_linear_regression)
        print('RMSE: %.3f TSSE: %.3f RSSE: %.3f ESSE: %.3f R^2: %.3f \n' % (res_slr[0], res_slr[1], res_slr[2], res_slr[3], res_slr[4]))
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('Grade')
        plt.ylabel('Prices')
        plt.title('Price vs. Grade')
        plt.grid(None, 'both', 'both')
    else:
        pass

# End of File