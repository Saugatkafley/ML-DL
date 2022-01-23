#Created on 21/04/2021
#@author saugatkafley

#Importing Libraires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state= 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors= 5 , metric ='minkowski', p=2)
classifier.fit(x_train,y_train)

print(classifier.predict(sc_x.transform([[30,87000]])))

y_pred = classifier.predict(x_test)
print( np.concatenate((y_pred.reshape(len(y_pred),1) , (y_test.reshape(len(y_test), 1))), axis=1))


from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
'''
from matplotlib.colors import ListedColormap
X_set, y_set = sc_x.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc_x.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''