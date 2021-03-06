# -*- coding: utf-8 -*-
"""
@author: ML-Group-6 --- Girish, Sashank, Ryan, Chengle
"""

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sklm
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

# Multiple Linear Regression on Dataset

# Load and Prepare Data
filename = 'house_data.csv'
dataset = load_csv(filename)

# Create linear regression object
regr = linear_model.LinearRegression()

def run_regr():
    evs_avg, mae_avg, mse_avg, rmse_avg, mdae_avg, rsq_avg = 0, 0, 0, 0, 0, 0
    for i in range(0, 5):
        print('Run : %d' % (i + 1))
        sets = split_5fold(dataset)
        test = sets.pop(i)
        sets = split_5fold(dataset)
        test = sets.pop(i)
        frames = [sets[0], sets[1], sets[2], sets[3]]
        train = pd.concat(frames)
        kc_house_x1_test = [row['bedrooms'] for index, row in test.iterrows()]
        global xs1
        xs1 = kc_house_x1_test
        kc_house_x2_test = [row['bathrooms'] for index, row in test.iterrows()]
        global xs2
        xs2 = kc_house_x2_test
        kc_house_x3_test = [row['sqft_living'] for index, row in test.iterrows()]
        global xs3
        xs3 = kc_house_x3_test
        kc_house_x4_test = [row['sqft_lot'] for index, row in test.iterrows()]
        global xs4
        xs4 = kc_house_x4_test
        kc_house_x5_test = [row['grade'] for index, row in test.iterrows()]
        global xs5
        xs5 = kc_house_x5_test
        kc_house_y_test = [row['price'] for index, row in test.iterrows()]
        global ys
        ys = kc_house_y_test
        test_set = [[row['bedrooms'], row['bathrooms'],
                     row['sqft_living'], row['sqft_lot'], row['grade']
                     ] for index, row in test.iterrows()]
        test_set = np.asarray(test_set)
        train_x = [[row['bedrooms'], row['bathrooms'],
                    row['sqft_living'], row['sqft_lot'], row['grade']
                    ] for index, row in train.iterrows()]
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