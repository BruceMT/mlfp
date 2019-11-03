import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

f=pd.read_csv("BCHI-dataset_2019-03-04.csv")

f=f[f.Indicator.isin(['Lung Cancer Mortality Rate (Age-Adjusted; Per 100,000 people)'])]
tset=f[["Year","Sex","Race/Ethnicity","Place"]].values
head=list(f.columns.values)
resualt=f["Value"].values

le = preprocessing.LabelEncoder()
for i in range(4):
    tset[:,i] = le.fit_transform(tset[:,i])

X_train, X_test, y_train, y_test = train_test_split(tset, resualt, test_size=0.30)


