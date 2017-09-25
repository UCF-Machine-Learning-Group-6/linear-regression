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

# Global Varibles
xs = list()
ys = list()
yhats = list()

# Simple Linear Regression on Dataset

# Load and Prepare Data
filename = 'house_data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

# Create linear regression object
regr = linear_model.LinearRegression()

def run_regr():
    evs_avg, mae_avg, mse_avg, rmse_avg, mdae_avg, rsq_avg = 0, 0, 0, 0, 0, 0
    for i in range(0, 5):
        print('Run : %d' % (i + 1))
        sets = split_5fold(dataset)
        test = sets.pop(i)
        kc_house_x_test = np.asarray([row[r] for row in test]).reshape(-1, 1)
        global xs
        xs = kc_house_x_test
        kc_house_y_test = np.asarray([row[6] for row in test]).reshape(-1, 1)
        global ys
        ys = kc_house_y_test
        train = sets[0] + sets[1] + sets[2] + sets[3]
        kc_house_x_train = np.asarray([row[r] for row in train]).reshape(-1, 1)
        kc_house_y_train = np.asarray([row[6] for row in train]).reshape(-1, 1)
        
        # Train the model using the training sets
        regr.fit(kc_house_x_train, kc_house_y_train)
        
        # Make predictions using the testing set
        kc_house_y_pred = regr.predict(kc_house_x_test)
        global yhats
        yhats = kc_house_y_pred
        
        # The Coefficients
        print('Coefficients : \n', regr.coef_)
        # Explained Variance Regression Score
        evs = float(sklm.explained_variance_score(kc_house_y_test, kc_house_y_pred))
        print('Explained Variance Regression Score : %.3f' % evs)
        # Mean Absolute Error
        mae = float(sklm.mean_absolute_error(kc_house_y_test, kc_house_y_pred))
        print('Mean Absolute Error : %.3f' % mae)
        # Mean Squared Error
        mse = float(sklm.mean_squared_error(kc_house_y_test, kc_house_y_pred))
        print('Mean Squared Error : %.3f' % mse)
        # Root Mean Squared Error
        rmse = float(sqrt(sklm.mean_squared_error(kc_house_y_test, kc_house_y_pred)))
        print('Root Mean Squared Error : %.3f' % rmse)
        # Median Absolute Error
        mdae = float(sklm.median_absolute_error(kc_house_y_test, kc_house_y_pred))
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
    
    print('Average Explained Variance Regression Score : %.3f' % float(evs_avg/5))
    print('Average Mean Absolute Error : %.3f' % float(mae_avg/5))
    print('Average Mean Squared Error : %.3f' % float(mse_avg/5))
    print('Average Root Mean Squared Error : %.3f' % float(rmse_avg/5))
    print('Average Median Absolute Error : %.3f' % float(mdae_avg/5))
    print('Average R^2 Regression Score : %.3f' % float(rsq_avg/5))
    print('\n')
    
    return None

# Evaluate Algorithm
for r in range(1,6):
    if r == 1:
        print('Price vs. Bedrooms \n')
        run_regr()
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('No. of Bedrooms')
        plt.ylabel('Prices')
        plt.title('Price vs. Bedrooms')
        plt.grid(None, 'both', 'both')
    elif r == 2:
        print('Price vs. Bathrooms \n')
        run_regr()
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('No. of Bathrooms')
        plt.ylabel('Prices')
        plt.title('Price vs. Bathrooms')
        plt.grid(None, 'both', 'both')
    elif r == 3:
        print('Price vs. Living SqFt \n')
        run_regr()
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('Living SqFt')
        plt.ylabel('Prices')
        plt.title('Price vs. Living SqFt')
        plt.grid(None, 'both', 'both')
    elif r == 4:
        print('Price vs. Lot SqFt \n')
        run_regr()
        plt.figure()
        plt.plot(xs, ys, 'b.')
        plt.plot(xs, yhats, 'r-')
        plt.xlabel('Lot SqFt')
        plt.ylabel('Prices')
        plt.title('Price vs. Lot SqFt')
        plt.grid(None, 'both', 'both')
    elif r == 5:
        print('Price vs. Grade \n')
        run_regr()
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