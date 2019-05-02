#!/usr/bin/env python
# coding: utf-8

# In[21]:

import pickle
import pandas as pd

sad_data=pd.read_csv('C:\\Users\\Bilal\\Desktop\\sad2.csv')
sad_data.head(10)
sad_data.isnull()
x=sad_data.drop("PredictedValue",axis=1)
y=sad_data["PredictedValue"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)

filename = 'pickleSadmodel.sav'
pickle.dump(logmodel, open(filename, 'wb'))
loaded_sad_model = pickle.load(open(filename, 'rb'))
result = loaded_sad_model.score(x_test, y_test)
print(result)

def predictedValue(dataList =[]):

    Xnew=[dataList]
    ynew = logmodel.predict(Xnew)
    return ynew




