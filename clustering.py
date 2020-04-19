import pandas as pd
import csv
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import random
import math
from mpl_toolkits import mplot3d

from os import system, name 
def clear(): 
    if name == 'nt': 
        _ = system('cls') 
    else: 
        _ = system('clear') 

def euclidianDistance(a,b):
    hasil = 0
    for i in range(len(a)):
        hasil = hasil + math.pow(a[i]-b[i],2)
    return math.sqrt(hasil)

def meanCentroid(c):
    new_centroid = []
    for i in range(len(c[0])):
        res = 0
        for j in range(len(c)):
            res = res + c[j][i]
        res = res/len(c)
        new_centroid.append(res)
    return new_centroid

def initiateCentroidData():
    c_arr = []
    for i in range(len(centroid)):
        c_arr.append([])
    return c_arr

def showResult(c_arr):
    for i in range(len(c_arr)):
        print('Kelas =',i)
        print(c_arr[i])

def getValidFeature(file):
    dataset = np.array(file)
    feature = []
    for i in range(len(dataset[1])):
        if isinstance(dataset[1][i], (int, float, complex)):
            feature.append(file.columns[i])
    return feature

def showFeature(features):
    for i in range(len(features)):
        print(i+1,features[i])

selected_feature = []
clear()
while True:
    fifa20 = pd.read_csv('fifa20.csv')
    X = np.array(fifa20)
    print('Selected Features',selected_feature)
    print('Minimum Two Features!')
    print('1. Select Feature')
    print('2. Reset Feature')
    print('3. Start')
    print('4. Exit')
    print('')
    print('Note: plot can be showed by only 2 or 3 features!')
    option = input('Pilih Opsi: ')
    if option == '1':
        clear()
        print('Selected Features',selected_feature)
        features = np.array(fifa20.columns)
        showFeature(features)
        select_feature = input('Select Featrue, ex (attacking_crossing) :')
        if select_feature in features:
            if select_feature in selected_feature:
                clear()
                print('Feature already selected!')
            else:
                selected_feature.append(select_feature)
                clear()
        else:
            clear()
            print('Feature not found!')
    if option == '2':
        clear()
        selected_feature = []
    if option == '3':
        clear()
        if len(selected_feature) < 2:
            print('Feature not enough!')
        else:
            fifa20 = pd.read_csv('fifa20.csv',usecols = selected_feature)
            fifa20 = fifa20.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) #hapus row jika terdapat fitur yang kosong

            for i in selected_feature:
                le = LabelEncoder()
                fifa20[i] = le.fit_transform(fifa20[i].astype(str))
            print(fifa20)
            if len(fifa20) == 0:
                clear()
                print('Cant clustering with this features ',selected_feature)
                continue
            k = input('Enter K (2-10) : ')
            try:
                k = int(k)
            except:
                clear()
                print('Cant clustering with this K')
                continue
            if not 2 <= k <= 10:
                clear()
                print('Cant clustering with this K')
                continue
            X = np.array(fifa20)
            centroid = random.sample(list(X), k)
            centroid = np.array(centroid)
            #array untuk data yang terdekat dengan centroid
            while True:
                c_arr = initiateCentroidData()
                #masukkan data ke centroid terdekat
                for i in range(len(X)):
                    idx_min = 0
                    min_dis = euclidianDistance(X[i],centroid[idx_min])
                    for j in range(1,len(centroid)):
                        temp_min = euclidianDistance(X[i],centroid[j])
                        if min_dis > temp_min:
                            idx_min = j
                            min_dis = temp_min
                    c_arr[idx_min].append(X[i])
                old_centroid = centroid.copy()
                for i in range(len(centroid)):
                    centroid[i] = meanCentroid(c_arr[i])
                if (old_centroid==centroid).all():
                    break
            try:
                with open('clustering_result.csv','wb') as file:
                    #write column
                    str_write = ''
                    for i in selected_feature:
                        str_write = str_write+i+','
                    file.write(str_write.encode())
                    file.write('label'.encode())
                    file.write('\n'.encode())
                    #write data
                    for i in range(len(c_arr)):
                        for j in c_arr[i]:
                            str_write = ''
                            for k in j:
                                str_write = str_write+str(k)+','
                            file.write(str_write.encode())
                            file.write(str(i).encode())
                            file.write('\n'.encode())
                print('result saved in clustering_result.csv')
            except:
                print('Result cant be saved!')

            if len(c_arr[0][0]) == 2:
                for i in c_arr:
                    plt.scatter([row[0] for row in i],[row[1] for row in i])
                for i in centroid:
                    plt.scatter(i[0],i[1],c='black')
                plt.xlabel(selected_feature[0])
                plt.ylabel(selected_feature[1])
                plt.show()
            if len(c_arr[0][0]) == 3:
                ax = plt.axes(projection="3d")
                for i in c_arr:
                    ax.scatter3D([row[0] for row in i],[row[1] for row in i],[row[2] for row in i])
                for i in centroid:
                    ax.scatter3D(i[0],i[1],i[2],c='black')
                ax.set_xlabel(selected_feature[0])
                ax.set_ylabel(selected_feature[1])
                ax.set_zlabel(selected_feature[2])  
                plt.show()
            
    if option == '4':
        break
a = input('Press any key to exit...')

            
