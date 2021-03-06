# -*- coding: utf-8 -*-
"""
@author: ML-Group-6 --- Girish, Sashank, Ryan, Chengle
"""

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load a CSV file
def load_csv(filename):
    dataset = pd.read_csv(filename, names = ['id', 'bedrooms', 'bathrooms', 
                                             'sqft_living', 'sqft_lot', 
                                             'grade', 'price'])
    return dataset

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
xs1 = list()
xs2 = list()
xs3 = list()
xs4 = list()
xs5 = list()
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
    rsq = 1 - (r / t)
    return rsq

# Calculate Root Mean Squared Error
def rmse(actual, predicted):
    sum_error = 0.0
    for i in range(len(ys)):
        prediction_error = np.subtract(yhats, ys).reshape((-1, 1))
        prediction_error = prediction_error ** 2
        sum_error = sum(prediction_error)
    mean_error = sum_error / float(len(ys))
    return sqrt(mean_error)

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, *args):
    rmse_avg, sst_avg, ssr_avg, sse_avg, r_sq_avg = 0, 0, 0, 0, 0
    for i in range(0,5):
        print('Run : %d' % (i + 1))
        sets = split_5fold(dataset)
        test = sets.pop(i)
        ytest = [row['price'] for index, row in test.iterrows()]
        global ys
        ys = ytest
        x1test = [row['bedrooms'] for index, row in test.iterrows()]
        global xs1
        xs1 = x1test
        x2test = [row['bathrooms'] for index, row in test.iterrows()]
        global xs2
        xs2 = x2test
        x3test = [row['sqft_living'] for index, row in test.iterrows()] 
        global xs3
        xs3 = x3test
        x4test = [row['sqft_lot'] for index, row in test.iterrows()]
        global xs4
        xs4 = x4test
        x5test = [row['grade'] for index, row in test.iterrows()]
        global xs5
        xs5 = x5test
        sets = [sets[0], sets[1], sets[2], sets[3]]
        train = pd.concat(sets)
        test_set = [[row['bedrooms'], row['bathrooms'], row['sqft_living'
                     ], row['sqft_lot'], row['grade'
                           ]] for index, row in test.iterrows()]
        predicted = algorithm(train, test_set, *args)
        actual = [row[-1] for row in test]
        rms = rmse(actual, predicted)
        rmse_avg += rms
        print('Root Mean Squared Error : %.3f' % float(rms))
        sst_avg += tsse(ys, ymean)
        print('Total Sum of Squared Error : %.3f' % float(tsse(ys, ymean)))
        ssr_avg += rsse(ys, yhats)
        print('Residual Sum of Squared Error : %.3f' % float(rsse(ys, yhats)))
        sse_avg += esse(yhats, ymean)
        print('Explained Sum of Squared Error : %.3f' % float(esse(yhats, ymean)))
        r_sq_avg += r_sq(rsse(ys, yhats), tsse(ys, ymean))
        print('R^2 Regression Score : %.3f' % float(r_sq(rsse(ys, yhats), tsse(ys, ymean))))
        print('\n')
    return [rmse_avg/5, sst_avg/5, ssr_avg/5, sse_avg/5, r_sq_avg/5]

# Calculate the Mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))
 
# Calculate the Variance of a list of numbers
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# Calculate Covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# dataset - id, bedrooms, bathrooms, sqft_living15, sqft_lot15, grade, price
# Calculate Coefficients
def multiple_coefficients(dataset):
    x1 = [row['bedrooms'] for index, row in dataset.iterrows()]  # bedrooms
    x2 = [row['bathrooms'] for index, row in dataset.iterrows()] # bathrooms
    x3 = [row['sqft_living'] for index, row in dataset.iterrows()] # sqft_living15
    x4 = [row['sqft_lot'] for index, row in dataset.iterrows()] # sqft_lot15
    x5 = [row['grade'] for index, row in dataset.iterrows()] # grade
    y = [row['price'] for index, row in dataset.iterrows()] # price
    x1_mean = mean(x1)
    x2_mean = mean(x2)
    x3_mean = mean(x3)
    x4_mean = mean(x4)
    x5_mean = mean(x5)
    y_mean = mean(y)
    global ymean
    ymean = y_mean
    eq1_coeffs = [variance(x1, x1_mean), covariance(x2, x2_mean, x1, x1_mean), covariance(
            x3, x3_mean, x1, x1_mean), covariance(x4, x4_mean, x1, x1_mean), covariance(
                    x5, x5_mean, x1, x1_mean)]
    eq2_coeffs = [covariance(x1, x1_mean, x2, x2_mean), variance(x2, x2_mean), covariance(
            x3, x3_mean, x2, x2_mean), covariance(x4, x4_mean, x2, x2_mean), covariance(
                    x5, x5_mean, x2, x2_mean)]
    eq3_coeffs = [covariance(x1, x1_mean, x3, x3_mean), covariance(x2, x2_mean, x3, x3_mean), variance(
            x3, x3_mean), covariance(x4, x4_mean, x3, x3_mean), covariance(
                    x5, x5_mean, x3, x3_mean)]
    eq4_coeffs = [covariance(x1, x1_mean, x4, x4_mean), covariance(x2, x2_mean, x4, x4_mean), covariance(
            x3, x3_mean, x4, x4_mean), variance(x4, x4_mean), covariance(
                    x5, x5_mean, x4, x4_mean)]
    eq5_coeffs = [covariance(x1, x1_mean, x5, x5_mean), covariance(x2, x2_mean, x5, x5_mean), covariance(
            x3, x3_mean, x5, x5_mean), covariance(x4, x4_mean, x5, x5_mean), variance(
                    x5, x5_mean)]
    eq_coeffs = np.array([eq1_coeffs, eq2_coeffs, eq3_coeffs, eq4_coeffs, eq5_coeffs])
    eq_ycovs = np.array([covariance(y, y_mean, x1, x1_mean), covariance(
            y, y_mean, x2, x2_mean), covariance(y, y_mean, x3, x3_mean), covariance(
                    y, y_mean, x4, x4_mean), covariance(y, y_mean, x5, x5_mean)])
    b_coeffs = np.linalg.solve(eq_coeffs, eq_ycovs)
    b5 = b_coeffs[4]
    b4 = b_coeffs[3]
    b3 = b_coeffs[2]
    b2 = b_coeffs[1]
    b1 = b_coeffs[0]
    b0 = y_mean - (b1*x1_mean + b2*x2_mean + b3*x3_mean + b4*x4_mean + b5*x5_mean)
    return [b0, b1, b2, b3, b4, b5]

# Multiple Linear Regression Algorithm
def multiple_linear_regression(train, test):
    predictions = list()
    b0, b1, b2, b3, b4, b5 = multiple_coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0] + b2 * row[1] + b3 * row[2] + b4 * row[3] + b5 * row[4]
        predictions.append(yhat)
    global yhats
    yhats = predictions
    return predictions

# Multiple Linear Regression on Dataset

# Load and Prepare Data
filename = 'house_data.csv'
dataset = load_csv(filename)

def run_regr():
    res_mlr = evaluate_algorithm(dataset, multiple_linear_regression)
    print('Average Root Mean Squared Error : %.3f' % res_mlr[0])
    print('Average Total Sum of Squared Error : %.3f' % res_mlr[1])
    print('Average Residual Sum of Squared Error : %.3f' % res_mlr[2])
    print('Average Explained Sum of Squared Error : %.3f' % res_mlr[3])
    print('Average R^2 Regression Score : %.3f' % res_mlr[4])
    print('\n')
    return None

# Evaluate Algorithm
print('Price vs. Bedrooms, Bathrooms, Living Sqft, Lot SqFt, Grade \n')
run_regr()
plt.figure()
plt.plot(xs1, ys, 'b.')
plt.plot(xs1, yhats, 'r-')
plt.xlabel('No. of Bedrooms')
plt.ylabel('Prices')
plt.title('Price vs. Bedrooms')
plt.grid(None, 'both', 'both')
plt.figure()
plt.plot(xs2, ys, 'b.')
plt.plot(xs2, yhats, 'r-')
plt.xlabel('No. of Bathrooms')
plt.ylabel('Prices')
plt.title('Price vs. Bathrooms')
plt.grid(None, 'both', 'both')
plt.figure()
plt.plot(xs3, ys, 'b.')
plt.plot(xs3, yhats, 'r-')
plt.xlabel('Living SqFt')
plt.ylabel('Prices')
plt.title('Price vs. Living SqFt')
plt.grid(None, 'both', 'both')
plt.figure()
plt.plot(xs4, ys, 'b.')
plt.plot(xs4, yhats, 'r-')
plt.xlabel('Lot SqFt')
plt.ylabel('Prices')
plt.title('Price vs. Lot SqFt')
plt.grid(None, 'both', 'both')
plt.figure()
plt.plot(xs5, ys, 'b.')
plt.plot(xs5, yhats, 'r-')
plt.xlabel('Grade')
plt.ylabel('Prices')
plt.title('Price vs. Grade')
plt.grid(None, 'both', 'both')
plt.figure()
plt.plot(xs1, ys, 'b.')
plt.plot(xs1, yhats, 'r-')
plt.plot(xs2, ys, 'b.')
plt.plot(xs2, yhats, 'r-')
plt.plot(xs3, ys, 'b.')
plt.plot(xs3, yhats, 'r-')
plt.plot(xs4, ys, 'b.')
plt.plot(xs4, yhats, 'r-')
plt.plot(xs5, ys, 'b.')
plt.plot(xs5, yhats, 'r-')
plt.xlabel('Bedrooms, Bathrooms, Living SqFt, Lot SqFt, Grade')
plt.ylabel('Prices')
plt.title('Price vs. Bedrooms, Bathrooms, Living SqFt, Lot SqFt, Grade')
plt.grid(None, 'both', 'both')

# End of File