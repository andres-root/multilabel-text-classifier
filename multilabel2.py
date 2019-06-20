#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from pandas import DataFrame
import pandas as pd

column = ['Title','Body']
dfBA = DataFrame(columns=column)
dfT = DataFrame(columns=[0, 1, 2, 3, 4])


# In[2]:


def create_data_frame(data):
    t = {}
    d = {
        'Title': [data[0]],
        'Body': [data[1]],
    }

    for n in range(5):
        if len(data[2]) > n:
            t[n] = [data[2][n]]
        else:
            t[n] = ['0']
        
    df = DataFrame(data=d)

    global dfBA, dfT
    
    dfBA = dfBA.append(df)
    dfT = dfT.append(DataFrame(data=t))


# In[3]:


def parser(path):
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        tags = []

        for row in reader:
            tags = tuple([x for x in row[2:-1] if x != ''])
            terms = [row[0], row[1], tags]
            create_data_frame(terms)


# In[4]:


path = 'data/vzn/TrainingData.csv'
parser(path)
import pandas as pd
df = pd.concat([dfBA, dfT], axis=1)


# In[5]:


dfBA


# In[6]:


dfT


# In[7]:


from nltk.corpus import stopwords

stopWordList=stopwords.words('english')
stopWordList.remove('no')
stopWordList.remove('not')

import unicodedata
import spacy


def removeAscendingChar(data):
    data=unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return data

def removeCharDigit(text):
    str='`1234567890-=~@#$%^&*()_+[!{;":\'><.,/?"}]'
    for w in text:
        if w in str:
            text=text.replace(w,'')
    return text

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer


lemma=WordNetLemmatizer()
token=ToktokTokenizer()

def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w,'v')
        #print(x)
        listLemma.append(x)
    return text

def stopWordsRemove(text):
    
    wordList=[x.lower().strip() for x in token.tokenize(text)]
    
    removedList=[x for x in wordList if not x in stopWordList]
    text=' '.join(removedList)
    #print(text)
    return text

def PreProcessing(text):
    text = removeAscendingChar(text)
    text = removeCharDigit(text)
    text = lemitizeWords(text)
    text = stopWordsRemove(text)

    return text


# In[ ]:


totalText=''

for x in df['Body']:
    ps = PreProcessing(x)
    totalText = totalText + " " + ps


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


wc=WordCloud(max_font_size=60).generate(totalText)
plt.figure(figsize=(16,12))
plt.imshow(wc, interpolation="bilinear")


# In[ ]:


import nltk

freqdist = nltk.FreqDist(token.tokenize(totalText))
plt.figure(figsize=(16,5))
freqdist.plot(50)


# In[ ]:


totalText=''

for x in df['Title']:
    ps=PreProcessing(x)
    totalText=totalText+" "+ps


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc=WordCloud(max_font_size=60).generate(totalText)
plt.figure(figsize=(16,12))
plt.imshow(wc, interpolation="bilinear")


# In[ ]:


import nltk

freqdist = nltk.FreqDist(token.tokenize(totalText))
freqdist
plt.figure(figsize=(16,5))
freqdist.plot(50)


# In[ ]:


x = df.iloc[:, 0:2].values
y = df.iloc[:, 2:-1]


# In[ ]:


okList=[]
for cl in dfT.columns:
     for n in df[cl]:
            okList.append(n)
okList=list(set(okList))
del(okList[okList.index('0')])


# In[ ]:


print(okList)


# In[ ]:


newDF=DataFrame(columns=okList)


# In[ ]:


for x in range(dfT.count()[0]):
    someDict={}
    for d in okList:
        rowdata=list(dfT.iloc[x])
        if d in rowdata:
            someDict[d]=1
        else:
            someDict[d]=0
    newDF=newDF.append(someDict,ignore_index=True)


# In[ ]:


newDF


# In[ ]:


dfBA.index=range(dfBA.count()[0])
df=dfBA.join(newDF)


# In[ ]:


df


# ### Binary relevance

# In[168]:


x=df.iloc[:,0:2].values
y=df.iloc[:,2:-1].values
# using binary relevance
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
x1=df.Title
x2=df.Body
from pandas import DataFrame
cv=CountVectorizer().fit(x1)
header=DataFrame(cv.transform(x1).todense(),columns=cv.get_feature_names())
cvArticle=CountVectorizer().fit(x2)
article=DataFrame(cvArticle.transform(x2).todense(),columns=cvArticle.get_feature_names())
import pandas as pd
x=pd.concat([header,article],axis=1)


# In[169]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidfhead=TfidfTransformer().fit(header)
head=DataFrame(tfidfhead.transform(header).todense())
tfidfart=TfidfTransformer().fit(article)
art=DataFrame(tfidfart.transform(article).todense())
import pandas as pd
x=pd.concat([head,art],axis=1)


# In[170]:


# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(xtrain.astype(float), ytrain.astype(float))

predictions = classifier.predict(xtest.astype(float))
predictions.toarray()
from sklearn.metrics import accuracy_score
accuracy_score(ytest.astype(float),predictions)


# ### Classifier chains

# In[171]:


# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(xtrain.astype(float), ytrain.astype(float))

# predict
predictions = classifier.predict(xtest.astype(float))

accuracy_score(ytest.astype(float),predictions)


# ### Label powerset

# In[172]:


# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(xtrain.astype(float), ytrain.astype(float))
# predict
predictions = classifier.predict(xtest.astype(float))

accuracy_score(ytest.astype(float),predictions)


# ### MLkNN

# In[173]:


from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=7)

# train
classifier.fit(xtrain.astype(float), ytrain.astype(float))
# predict
predictions = classifier.predict(xtest.astype(float))

accuracy_score(ytest.astype(float),predictions)


# In[ ]:




