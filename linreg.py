import pandas as pd
import numpy as np

# Load a CSV file

# Ryan Fatt: Changed to use Pandas


def load_csv(filename):
    dataset = pd.read_csv(filename, names=[
                          'id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade'])
    return dataset

# Split a dataset into a train and test set

# Ryan Fatt: Added log transform for some features


def transform(dataset):
    dataset['price'] = dataset['price'].apply(np.log)
    dataset['sqft_lot'] = dataset['sqft_lot'].apply(np.log)
    return dataset

# Changed to use numpy since the code is way simpler


def train_test_split(dataset):
    train, validate, test = np.split(dataset.sample(
        frac=1), [int(.6 * len(dataset)), int(.8 * len(dataset))])
    return train, test

# Calculate root mean squared error

# Ryan Fatt: Changed to use numpy square root


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        for prediction in prediction_error:
            sum_error += (prediction ** 2)
    mean_error = sum_error / float(len(actual))
    return np.sqrt(mean_error)

# Evaluate an algorithm using a train/test split


def evaluate_algorithm(dataset, algorithm, *args):
    train, test = train_test_split(dataset)
    predicted = algorithm(train, test, *args)
    actual = []
    for index, row in test.iterrows():
        actual.append(row['price'])
    rmse = rmse_metric(actual, predicted)
    return rmse

# Calculate covariance between x and y


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# Calculate the variance of a list of numbers


def variance(values, mean):
    return sum([(x - mean)**2 for x in values])

# Calculate coefficients

# Ryan Fatt: Made Multifeature and to use Pandas and Numpy


def coefficients(dataset, test):
    x = []
    y = []
    for index, row in dataset.iterrows():
        x.append([row['bedrooms'], row["bathrooms"],
                  row["sqft_living"], row["sqft_lot"]])
        y.append(row['price'])
    x_mean, y_mean = np.mean(x, axis=0), np.mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

# Simple linear regression algorithm

# Ryan Fatt: Made multifeature


def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train, test)
    for index, row in test.iterrows():
        yhat = b0 + b1 * [row['bedrooms'], row["bathrooms"],
                          row["sqft_living"], row["sqft_lot"]]
        predictions.append(yhat)
    return predictions

# Simple linear regression on dataset
# load and prepare data
filename = 'kc_house_data_pruned_nohead.csv'
dataset = load_csv(filename)
transformed_dataset = transform(dataset)
# evaluate algorithm
rmse = evaluate_algorithm(transformed_dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))
