import time
import pickle
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
    filename = 'pickleSadmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    list2 = [[]]
    temp = [0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 13]
    for i in range(len(temp)):
        list2[0].append(list[i])

    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://cerebrex-101.firebaseio.com', None)
    update = firebase.put('/Results/-LddRqXtD2qSyzoEekFs', "prediction", str(loaded_model.predict(list2)[0]))   #updating a value under a key

    result = firebase.get('/Results', '-LddRqXtD2qSyzoEekFs')

    return result

def frontEndSkip(list):
    filename = 'pickleSkipmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    list2 = [[]]
    temp = [0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 13]
    for i in range(len(temp)):
        list2[0].append(list[i])

    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://cerebrex-101.firebaseio.com', None)
    update = firebase.put('/Skip/-LdrO5Oc8GLRupBSiKqL', "prediction", str(loaded_model.predict(list2)[0]))

    result = firebase.get('/Skip', '-LdrO5Oc8GLRupBSiKqL')

    return result

def frontEndStress(list):
    filename = 'pickleStressmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    list1 = [[]]
    for i in range(len(list)):
        list1[0].append(list[i])
    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://cerebrex-101.firebaseio.com', None)

    update = firebase.put('/Stress/-LdrScJCDAWx5idPslrB', "predictionStress", str(loaded_model.predict(list1)[0]))

    result = firebase.get('/Stress', '-LdrScJCDAWx5idPslrB')

    return result
while True:
    start = time.time()
    print('sad',frontEndSad(backEnd()))
    print('skip',frontEndSkip(backEnd()))
    print('stress', frontEndStress(backEnd()))
    end = time.time()
    print("Time taken :", end - start)
