#!/usr/bin/env python
# coding: utf-8

# In[21]:

import pickle
import pandas as pd

stress_data=pd.read_csv('C:\\Users\\Bilal\\Desktop\\stress.csv')
stress_data.head(10)
stress_data.isnull()
x=stress_data.drop("PredictedValue",axis=1)
y=stress_data["PredictedValue"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)

filename = 'pickleStressmodel.sav'
pickle.dump(logmodel, open(filename, 'wb'))
loaded_sad_model = pickle.load(open(filename, 'rb'))
result = loaded_sad_model.score(x_test, y_test)
print(result)
# from sklearn.metrics import classification_report
#
# classification_report( y_test,predictions)
#
# from sklearn.metrics import confusion_matrix
#
# confusion_matrix(y_test,predictions)
#
# from sklearn.metrics import accuracy_score
#
# accuracy_score(y_test,predictions)  #The piece of code that calculates the accuracy
# # Xnew = [[4073.333252,	4143.07666,	4048.717773,	4100.512695,	4137.94873,	4192.307617,	4106.666504,	4145.12793,	4174.358887	,4154.358887,	4101.538574,	4024.102539	,4151.794922,	4124.102539]]
# score = logmodel.score(x_test, y_test)
# print("Accuracy of stress model :",score*100)

def stressValue(dataList =[]):

    Xnew=[dataList]
    ynew = logmodel.predict(Xnew)
    return ynew




