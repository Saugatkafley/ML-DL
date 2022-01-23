# Natural Language Processing

#%%# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#%% Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
#print(corpus)

#%% Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
#print(len(X[0]))

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

#%% Training the SVC on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators =10)
classifier.fit(X_train,y_train)
#%% Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print(tn, fp, fn, tp)
print("Accuracy  = ",(tp+tn)/(tp+tn+fp+fn))
print("Precision = ",(tp)/(tp+fp))
print("Recall    = ",(tp)/(tp+fn))
#%% Predicting entire new review
new_review = input()
'''
new_review = re.sub('[^a-zA-Z]',' ' , new_review)
new_review= new_review.lower()
new_revire = new_review.split()
new_ps = PorterStemmer()
new_review = [new_ps.stem(words) for words  in review if words not in set(stopwords.words('english')) ]
new_review = ''.join(new_review)
new_corpus = [new_review]
'''
new_Xtest =cv.transform([new_review]).toarray()
y_new = classifier.predict(new_Xtest)
print(y_new)
if y_new == [1]:
    print('Wow! This is good review. Our model has predicted it and its sure by...')
else:
    print('Sorry , This is a bad review. Our model has predicted it and its sure by...')
print(accuracy_score(y_test, y_pred) * 100, '%')
print("Accuracy  = ",(tp+tn)/(tp+tn+fp+fn))
print("Precision = ",(tp)/(tp+fp))
print("Recall    = ",(tp)/(tp+fn))
