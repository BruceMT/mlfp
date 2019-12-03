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

def pl(str):
    plt.figure(str)
    r1los = r1[r1[3].isin([str])].values
    plt.plot(r1los[:, 0], r1los[:, -1], 'r')
    plt.plot(r1los[:, 0], r1los[:, -2], 'b')
    plt.show()

f=pd.read_csv("processed.csv").values
row=pd.read_csv("row.csv").values[:,1:]


dic=rdic()

data=f[:,1:]
data[:,0]=data[:,0]-2009
resualt=f[:,0]
resualt=np.reshape(resualt,(resualt.size,1))



lr=NN([[5],[1,'relu']])
lr.fit(data,resualt,epochs=300,learning_rate=0.1)
res1=lr.predict(data)



r1=pd.DataFrame(row)
r1.insert(5,"res",res1)
r1.to_csv()
for k in dic.keys():
    pl(k)


#ann

ann=NN([[5],[7,'sigmoid'],[8,'relu'],[4,'relu'],[1,'relu']])
ann.fit(data,resualt,epochs=300)
res2=lr.predict(data)

r2=pd.DataFrame(row)
r2.insert(5,"res",res1)
r2.to_csv()
for k in dic.keys():
    pl(k)






