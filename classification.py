import pandas as pd
import csv
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from os import system, name 
def clear(): 
    if name == 'nt': 
        _ = system('cls') 
    else: 
        _ = system('clear') 
def getValidFeature(file):
    fifa20 = pd.read_csv(file)
    dataset = np.array(fifa20)
    feature = []
    for i in range(len(dataset[1])):
        if isinstance(dataset[1][i], (int, float, complex)):
            feature.append(fifa20.columns[i])
    return feature

def getValidLabel(file):
    fifa20 = pd.read_csv(file)
    dataset = np.array(fifa20)
    label = []
    for i in range(len(dataset[1])):
        if not len(dataset[:,i]) == len(set(dataset[:,i])):
            label.append(fifa20.columns[i])
    return label
def showFeature(features):
    for i in range(len(features)):
        print(i+1,features[i])

def startClassification(label, features, file):
    try:
            if not len(features) == 0:
                useCol = features.copy()
                useCol.append(label)
                fifa20 = pd.read_csv(file, usecols = useCol)
            else:
                fifa20 = pd.read_csv(file)
            fifa20 = fifa20.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) #hapus row jika terdapat fitur yang kosong
    except:
        print('File Clustering Not Found!')
        exit()
    print(fifa20)
    le = LabelEncoder()
    fifa20[label] = le.fit_transform(fifa20[label].astype(str))
    X = np.array(fifa20.drop([label], axis=1))
    y = np.array(fifa20[label])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2)
    models = [
        ('Linear', SGDClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('SVM', SVC(gamma='auto')),
        ('K-Neighbors', KNeighborsClassifier()),
        ('Decision Tree', DecisionTreeClassifier()),
    ]
    for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        svm_1_prediction = clf.predict(X_test)
        print(name)
        print (classification_report(y_test, svm_1_prediction,zero_division=1))
        print('')
    Pause = input('Press Any Key to Continue...')
loop = True
while loop:
    clear()
    print('1. Classification Clustering Result')
    print('2. Classification Manual')
    print('3. Exit')
    opsi = input('Pilih Opsi: ')
    if opsi == '1':
        startClassification('label',[],'clustering_result.csv')
    if opsi == '2':
        opsi_2_loop = True
        features = []
        labels = ''
        while opsi_2_loop:
            clear()
            print('Features: ',features)
            print('Labels: ',labels)
            print('1. Select Feature (Select two or more features)')
            print('2. Select Label (Only one label can be selected)')
            print('3. Reset Features and Label')
            print('4. Start')
            print('5. Back')
            opsi_2 = input('Pilih Opsi: ')
            if opsi_2 == '1':
                clear()
                features_csv = getValidFeature('fifa20.csv')
                showFeature(features_csv)
                select_feature = input('Select Featrue, ex (attacking_crossing) :')
                if select_feature in features_csv:
                    if select_feature in features:
                        clear()
                        print('Feature already selected!')
                    else:
                        features.append(select_feature)
                        clear()
                else:
                    clear()
                    print('Feature not found!')
                
            if opsi_2 == '2':
                if labels == '':
                    clear()
                    print('Please wait, getting label that can be use...')
                    label_csv = getValidLabel('fifa20.csv')
                    clear()
                    print(label_csv)
                    select_label = input('Select Label, ex (team_position) :')
                    if select_label in label_csv:
                        if select_label == labels:
                            clear()
                            print('Label already selected!')
                        else:
                            labels = select_label
                            clear()
                    else:
                        clear()
                        print('Label not found!')
                else:
                    print('Label already selected!')
            if opsi_2 == '3':
                features = []
                labels = ''
            if opsi_2 == '4':
                if len(features) >= 2:
                    if not labels == '':
                        startClassification(labels,features,'fifa20.csv')
            if opsi_2 == '5':
                opsi_2_loop = False
    if opsi == '3':
        exit()