import pandas as pd
import numpy as np
from mxnn import NN


f=pd.read_csv("processed.csv").values

data=f[:,1:]
data[:,0]=data[:,0]-2009
resualt=f[:,0]
resualt=np.reshape(resualt,(resualt.size,1))

#线性回归

lr=NN([[5],[1,'relu']])
lr.fit(data,resualt,epochs=300)
res1=lr.predict(data)

print (res)


#ann

ann=NN([[5],[7,'sigmoid'],[3,'relu'],[1,'relu']])
ann.fit(data,resualt,epoch=300)
res2=lr.predict(data)