# -*- coding:utf-8 -*- 
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import json

from geopy.geocoders import Nominatim
#import tensorflow as tf
#from mx import bpnn

geoneed=True #False

def rdic():
    f=open("dic","r")
    jf=f.read()
    data=dict()
    data=eval(jf)
    data['U.S. Total, U.S. Total']=[0.0,0.0]
    #print(data)
    f.close()
    return data


def sdic(dic):
    f=open("dic","w+")
    f.write(str(dic))
    f.close()

f=pd.read_csv("BCHI-dataset_2019-03-04.csv")

f=f[f.Indicator.isin(['Lung Cancer Mortality Rate (Age-Adjusted; Per 100,000 people)'])]
tset=f[["Year","Sex","Race/Ethnicity","Place","Value"]].values
head=list(f.columns.values)
resualt=f["Value"].values
place=f[["Place"]].values




loc=rdic()
#loc=dict();

#bpnn(sizes=[4,3,1])
#bpnn.SGD(bpnn,X_train,20,5,0.0001)
if geoneed:
    lc=list()
    gps = Nominatim()
    for address in place:

        asdf=loc.get(address[0])
        if asdf is None:
            location = gps.geocode(address)
            if location is not None:
                print(address,location.latitude,location.longitude)
                loc[address[0]]=[location.latitude,location.longitude]
                sdic(loc)
                lc.append()
        else:
            print(address,asdf[0],asdf[1])
            lc.append([asdf[0],asdf[1]])


for i in tset:
    num=loc.get(i[-2])
    i[-2]=num[0]
    i[-1]=num[1]

le = preprocessing.LabelEncoder()
for i in range(4):
    tset[:,i] = le.fit_transform(tset[:,i])
    

X_train, X_test, y_train, y_test = train_test_split(tset, resualt, test_size=0.30)

