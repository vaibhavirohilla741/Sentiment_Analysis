#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import re
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegressionCV



df=pd.read_csv('F:\COLLEGE\ML Project\sentiment analysis\movie_data.csv')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    return text

df['review'] = df['review'].apply(preprocessor)
porter=PorterStemmer()

def tokenizer(text):
     return text.split()

def tokenizer(text):
     return text.split()

def tokenizer_porter(text):
     return [porter.stem(word) for word in text.split()]
stop=stopwords.words('english')

tfidf = TfidfVectorizer(strip_accents=None,
                                        lowercase=False,
                                        preprocessor=None,
                                        tokenizer=tokenizer_porter,
                                        use_idf=True,
                                        norm='l2',
                                        smooth_idf=True)
y=df.sentiment.values
X=tfidf.fit_transform(df.review)

X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=1, test_size=0.5, shuffle=False)

clf=LogisticRegressionCV(cv=5,
                                             scoring='accuracy',
                                             random_state=0,
                                             n_jobs=-1,
                                             verbose=3,
                                             max_iter=300).fit(X_train,y_train)
saved_model=open('saved_model.sav','wb')
pickle.dump(clf, saved_model)
saved_model.close()

filename='saved_model.sav'
saved_clf=pickle.load(open(filename,'rb'))
print(saved_clf.score(X_test, y_test))























