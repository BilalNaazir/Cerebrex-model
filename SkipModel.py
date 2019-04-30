#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd

skip_data=pd.read_csv('C:\\Users\\Bilal\\Desktop\\skip3.csv')
skip_data.head(10)
skip_data.isnull()
x=skip_data.drop("PredictedValue",axis=1)
y=skip_data["PredictedValue"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)

from sklearn.metrics import classification_report

classification_report( y_test,predictions)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)
score = logmodel.score(x_test, y_test)
print("Accuracy of skip model :",score*100)

def skipValue(dataList =[]):

    Xnew=[dataList]
    ynew = logmodel.predict(Xnew)
    return ynew




print('s')