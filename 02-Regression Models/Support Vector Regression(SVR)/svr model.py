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
#reshape y 

y= y.reshape(len(y) , 1)

#feature Scaling 

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x= sc_x.fit_transform(x)
y= sc_y.fit_transform(y)


#Training Linear Model
from sklearn.svm import SVR
regressor = SVR(kernel ='rbf' , degree =4)
regressor.fit(x,y)

#Visualizing Support Vector Machine 

plt.scatter(sc_x.inverse_transform(x) , sc_y.inverse_transform(y) , color = 'red')
plt.plot(sc_x.inverse_transform(x) ,sc_y.inverse_transform(regressor.predict(x)) )
plt.title("Support Vector Machine")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

## Predicting Single Value 

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.00]]))))