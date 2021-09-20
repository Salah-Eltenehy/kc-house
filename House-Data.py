#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns

#Read data
data_set = pd.read_csv('kc_house_data.csv')

print(data_set.head())
print('*'*70)
print(data_set.describe())
print('*'*70)
print('Rows x Columns')
print(data_set.shape)
print('*'*70)
print(data_set.info())
print('*'*70)
sns.displot(data_set['price'])

#Independent & Dependent variables
X = data_set.iloc[:, 3:]
y = data_set.iloc[:, 2:3]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
#Spliting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size= 0.25, 
                                                    random_state= 0)
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators= 1000, random_state= 0, alpha= 0.9)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred2 = regressor.predict(X_train)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)

r2_train = r2_score(y_train, y_pred2)
mse_train = mean_squared_error(y_train, y_pred2)
mae_train = mean_absolute_error(y_train, y_pred2)


print('*'*50)
print('Gradient Boosting Results')
print('*'*50)
print('Training Set:')
print('R-Squared Value: ', r2_train)
print('Mean Squred Error: ', mse_train)
print('Mean Absolute Error: ', mae_train)
print('*'*50)
print('Test Set:')
print('R-Squared Value: ', r2_test)
print('Mean Squred Error: ', mse_test)
print('Mean Absolute Error: ', mae_test)














