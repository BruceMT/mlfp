import pandas as pd
import numpy as np
from mxnn import NN


f=pd.read_csv("processed.csv").values

data=f[:,1:]
data[:,0]=data[:,0]-2009
resualt=f[:,0]
resualt=np.reshape(resualt,(resualt.size,1))

lr=NN([[5],[1,'relu']])
lr.fit(data,resualt,epochs=300)
res=lr.predict(data)

print (res)