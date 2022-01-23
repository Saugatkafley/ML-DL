#Created on 21/04/2021
#@author saugatkafley

#Importing Libraires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Dataset

dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:-1].values
y= dataset.iloc[:,-1].values

#No need of splitting

# Training of entire model with Random FOrest Regression

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=1)
regressor.fit(x,y)

#Predicting values

print(regressor.predict([[6.0]]))

#Visualizing Data

X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()