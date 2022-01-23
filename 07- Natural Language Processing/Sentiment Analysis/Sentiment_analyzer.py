#%%import Libraries
import pandas as pd
from numba import jit,cuda

#%% Import dataset
dataset = pd.read_csv('train.csv' )
dataset_test = pd.read_csv('test.csv')
#%% Cleaning the Dataset
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus =[]
for i in range(0,7920):
    review = re.sub('[^a-zA-z]', " ",dataset['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(words) for  words in review if not words in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
#print(corpus)

#%% Creating Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=20343)
X_train = cv.fit_transform(corpus).toarray()
y_train= dataset.iloc[:,-2]
X_test = dataset_test.iloc[:,-1].values
#%%Trainign Model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
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
new_review = input("ENter a custom text")
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



