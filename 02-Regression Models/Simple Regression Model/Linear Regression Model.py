#Created on 19/04/2021
#Author saugatkafley

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset

dataset = pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

#split dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state =0)

#Model Training with Linear Regression

from sklearn.linear_model import LinearRegression

regressor =LinearRegression()

regressor.fit(x_train,y_train)

# Predicting Salary

y_predict = regressor.predict(x_train) 
#visualizing training data 

plt.scatter(x_train,y_train,color = 'r')
plt.plot(x_train,y_predict, color = 'blue')
plt.xlabel('Experience(t)')
plt.ylabel('Salary($)')
plt.title("Salary vs Experience (Training)")
plt.show()

#visualizing test data 

plt.scatter(x_test,y_test,color = 'r')
plt.plot(x_train,y_predict, color = 'blue')
plt.xlabel('Experience(t)')
plt.ylabel('Salary($)')
plt.title("Salary vs Experience(Test)")
plt.show()