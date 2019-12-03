import pandas as pd
import numpy as np
from mxnn import NN
import matplotlib.pyplot as plt

def rdic():
    f=open("dic","r")
    jf=f.read()
    data=dict()
    data=eval(jf)
    data['U.S. Total, U.S. Total']=[36.4280088,-92.0269643]
    #print(data)
    f.close()
    return data

f=pd.read_csv("processed.csv").values
row=pd.read_csv("row.csv").values


dic=rdic()

data=f[:,1:]
data[:,0]=data[:,0]-2009
resualt=f[:,0]
resualt=np.reshape(resualt,(resualt.size,1))



lr=NN([[5],[1,'relu']])
lr.fit(data,resualt,epochs=300,learning_rate=0.01)
res1=lr.predict(data)

print (res1)






#ann
"""
ann=NN([[5],[7,'sigmoid'],[8,'relu'],[4,'relu'],[1,'relu']])
ann.fit(data,resualt,epochs=300)
res2=lr.predict(data)

print(res2)

i=1
"""


