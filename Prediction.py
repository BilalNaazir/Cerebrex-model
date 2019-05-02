from SadModel import predictedValue
from SkipModel import skipValue
from StressModel import stressValue
import time
#

def backEnd():
    from firebase import firebase   #Connecting to the backend's firebase containing live raw data
    firebase = firebase.FirebaseApplication('https://cerebrex-c0a9a.firebaseio.com', None)
    result = firebase.get('/eeg', 'eegdata')    #obtaining raw data into a list
    result = result.split(",")   #splitting a string where it consists a comma to make a list
    list=[]
    for i in range(len(result)):
        list.append(float(result[i]))

    return list


def frontEndSad(list):
    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://cerebrex-101.firebaseio.com', None)
    update = firebase.put('/Results/-LddRqXtD2qSyzoEekFs', "prediction", str(predictedValue(list)[0]))   #updating a value under a key

    result = firebase.get('/Results', '-LddRqXtD2qSyzoEekFs')

    return result

def frontEndSkip(list):
    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://cerebrex-101.firebaseio.com', None)
    update = firebase.put('/Results/-LddUCH9tsUnnecsLgWs', "predictionSkip", str(skipValue(list)[0]))

    result = firebase.get('/Results', '-LddUCH9tsUnnecsLgWs')

    return result

def frontEndStress():
    list =[4073.333252,	4143.07666,	4048.717773, 4100.512695, 4137.94873,	 4192.307617, 4106.666504, 4145.12793, 4174.358887,4154.358887, 4101.538574, 4024.102539, 4151.794922, 4124.102539]
    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://cerebrex-101.firebaseio.com', None)
    update = firebase.put('/Results/-LdjktIjCZZUtr2wyqa8', "predictionStress", str(stressValue(list)[0]))

    result = firebase.get('/Results', '-LdjktIjCZZUtr2wyqa8')

    return result
while True:
    start = time.time()
    print(frontEndSad(backEnd()))
    print(frontEndSkip(backEnd()))
    # print(frontEndStress())
    end = time.time()
    print("Time taken :", end - start)





