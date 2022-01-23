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

#Training Linear Model

from sklearn.linear_model import LinearRegression
reg1 = LinearRegression()
reg1.fit(x,y)

#Now Training Polynomial Model

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures( degree = 6)
x_poly = poly_reg.fit_transform(x)

reg2 = LinearRegression()
reg2.fit(x_poly , y)

# Visualizing from Linear Model

plt.scatter(x,y ,color = 'red')
plt.plot(x,reg1.predict(x))
plt.title("Truth or Bluff")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Visualizing from Linear Polynomial Model

plt.scatter(x,y ,color = 'red')
plt.plot(x,reg2.predict(x_poly))
plt.title("Truth or Bluff")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Predicting single values from Linear model

print(reg1.predict([[6]]))


# Predicting single values from Linear Poly model

print(reg2.predict( poly_reg.fit_transform([[6]])))