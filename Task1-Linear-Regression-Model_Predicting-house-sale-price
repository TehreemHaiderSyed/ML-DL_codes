import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Dell\Downloads\ProdigyMaterial\Task1-Linear-Regression\train.csv")

selectedFeatures = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr', 'BsmtFullBath', 
                   'BsmtHalfBath ', 'FullBath', 'HalfBath']

data['SquareFootage'] = (data['1stFlrSF'] + data['2ndFlrSF'] )

data['Bathrooms'] =(data['BsmtFullBath'] + data['BsmtHalfBath'] + data['FullBath'] + data['HalfBath'])

data['BedroomsAboveGround'] = data['BedroomAbvGr']


Xtrain = data[['SquareFootage', 'Bathrooms', 'BedroomsAboveGround']].values
ytrain = data['SalePrice'].values

Xtrain_mean = np.mean(Xtrain, axis=0) #considering columns 
Xtrain_std = np.std(Xtrain, axis=0)
Xtrain = (Xtrain - Xtrain_mean) / Xtrain_std
Xtrain = np.hstack([np.ones((Xtrain.shape[0], 1)), Xtrain])

weights_pred = np.zeros(Xtrain.shape[1])
alpha = 1
iterations = 10

def cost_finding_function(X, y, weights_pred):
   
    predictions = np.dot(X, weights_pred)
    predictedcost = np.mean((predictions - y) )
    return predictedcost

def Optimizing_parameters(X, y, weights_pred, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = np.dot(X, weights_pred)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / m
        weights_pred -= alpha * gradient
        predictedcost = cost_finding_function(X, y, weights_pred)
        cost_history.append(predictedcost)

    return weights_pred, cost_history



weights_pred, cost_history = Optimizing_parameters(Xtrain, ytrain, weights_pred, alpha, iterations)

test_data = pd.read_csv(r"C:\Users\Dell\Downloads\ProdigyMaterial\Task1-Linear-Regression\test.csv")

test_data['SquareFootage'] = (test_data['1stFlrSF'] + test_data['2ndFlrSF']) #selected features to improve model's performance


test_data['Bathrooms'] = (test_data['BsmtFullBath'] + test_data['BsmtHalfBath'] + test_data['FullBath'] + 
                          test_data['HalfBath'])

test_data['BedroomsAboveGround'] = test_data['BedroomAbvGr']

Xtest = test_data[['SquareFootage', 'Bathrooms', 'BedroomsAboveGround']].values

Xtest = (Xtest - Xtrain_mean) / Xtrain_std
Xtest = np.hstack([np.ones((Xtest.shape[0], 1)), Xtest])

predicted_prices = Xtest @ weights_pred

for index, row in test_data.iterrows():
    sq_ft = row['SquareFootage']
    bedrooms = row['BedroomsAboveGround']
    bathrooms = row['Bathrooms']
    pred_price = predicted_prices[index]
    print(f'Sq_Ft: {sq_ft}, Beds: {bedrooms}, Baths: {bathrooms}, Pred. Sale Price: {pred_price}')

predicted_prices = predicted_prices[~np.isnan(predicted_prices)]

# Starndard values 
standard_values = pd.read_csv(r"C:\Users\Dell\Downloads\ProdigyMaterial\Task1-Linear-Regression\sample_submission.csv", usecols=[1]).values.flatten()
standard_values = standard_values[:len(predicted_prices)]  # Ensure same length

# Calculate Mean Absolute Error (MAE) and model's prediction accuracy 
mn_ab_er = mean_absolute_error(standard_values, predicted_prices)
mean_standard_values = np.mean(standard_values)
accuracy = (1 - (mn_ab_er / mean_standard_values)) * 100

print(f"Prediction Accuracy: {accuracy:.2f}%")

Houses = pd.DataFrame({'SerialNumber': range(1, 1458)})
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 7))

a.scatter(Houses, predicted_prices, color='blue', label='Predicted')
a.set_xlabel('House index')
a.set_ylabel('Predicted prices')
a.set_title('Predicted prices')
a.legend()
a.grid(True)

b.scatter(Houses, standard_values, color='orange', label='Standard')
b.set_xlabel('House index')
b.set_ylabel('Standard prices')
b.set_title('Standard prices (given sample)')
b.legend()
b.grid(True)
plt.tight_layout()
plt.show()

