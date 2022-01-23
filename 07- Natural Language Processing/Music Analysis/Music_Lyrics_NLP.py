#%%import Libraries
import pandas as pd

#%% Import dataset
dataset = pd.read_csv('Music_Analysis.csv' )
X = dataset.iloc[:,5].values
y = dataset.iloc[:,-1].values
print(dataset.head())
#%% Cleaning the Dataset
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

corpus =[]
for i in range(0,28373):
    review = re.sub(['^a-zA-Z'], " ",dataset['lyrics'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(words) for  words in review if not words in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
print(corpus)

#%% Creating Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
