
# coding=utf-8
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tanh(x) :
    return np.tanh(x)


def tanh_deriv(x) :
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistics(x) :
    return 1 / (1+np.exp(-x))


def logistics_deriv(x) :
    return logistics(x)*(1-logistics(x))


def sigmoid(x) :
    x=np.cast[float](x)
    return 1/(1+np.exp(-x))


def sigmoid_deriv(x) :
    return sigmoid(x)*(1-sigmoid(x))

def relu(signal):
    return np.maximum(0.1*signal, signal)


def relu_deriv(signal):
    return -np.minimum(0.1*signal, signal)


class NN :

    activation = []
    activation_deriv = []
    weight = []
    bp = []  # 存偏导数
    state = []
    dw = []
    outrange=1

    p=0
    r=0


    def __init__(self, layers, activation='sigmoid', inw=[-0.1,0.1]) :
        
        self.weight = [] 
        self.state.append([1]*layers[0])
        for i in range(1, len(layers)) :
            self.state.append([1]*layers[i])
            self.weight.append((2*np.random.random((layers[i], layers[i-1]+1))-1)*0.1)
            self.dw.append(np.zeros((layers[i], layers[i-1]+1)))
            self.bp.append([1]*(layers[i]))
            if activation[i-1] == 'logistic' :
                self.activation.append(logistics)
                self.activation_deriv.append(logistics_deriv)
            elif activation[i-1] == 'tanh' :
                self.activation.append( tanh)
                self.activation_deriv.append( tanh_deriv)
            elif activation[i-1] == 'sigmoid' :
                self.activation.append(sigmoid)
                self.activation_deriv.append( sigmoid_deriv)
            elif activation[i-1] == 'relu' :
                self.activation.append(relu)
                self.activation_deriv.append( relu_deriv)
        self.weight=np.array(self.weight)
        self.bp=np.array(self.bp)
        self.dw=np.array(self.dw)
        self.state=np.array(self.state)

    def fit(self, X, Y, learning_rate=0.5, epochs=10000, solver="SGD",eta=0.9, momentum=0.2, batch_size=32,cellrange=1) : #self nerual network forward&&back
        X = np.array(X)
        Y = np.array(Y)
        X=np.cast[float](X)
        self.outrange=cellrange

        for k in range(epochs) :
            tsta=self.state.copy()
            i = np.random.randint(X.shape[0])
            if solver=="SGD" :
                ran=np.random.randint(0, high=len(X), size=2, dtype='l')
                ran.sort()
                while not ran[1]-ran[0]>batch_size :
                    ran=np.random.randint(0, high=len(X), size=2, dtype='l')
                    ran.sort()
                tsta[0]=X[ran[0] :ran[1]]
                res=Y[ran[0] :ran[1]]
            else :
                tsta[0]=X.copy()
                res=Y.copy()
            #fw
            for i in range(len(self.weight)) :
                tsta[i]=np.insert(tsta[i], 0, values=np.ones(len(tsta[i])), axis=1)    
                tsta[i+1]=self.activation[i]( tsta[i].dot(self.weight[i].T) )*cellrange
                
            tsta1=tsta.copy()
            #bp
            for i in range(len(self.weight)-1,-1,-1)  :
                if i ==len(self.weight)-1 :
                    tsta[-1]=self.activation_deriv[i](tsta[-1])*(res-tsta[-1])#tsta[-1]*(cellrange-tsta[-1])*(Y[ran[0] :ran[1]]-tsta[-1])
                    tp=tsta[-1][0]
                    for j in tsta[-1]  :
                        tp=tp+j
                    tsta[-1]=tp/len(tsta[-1])
                    self.dw[i]=momentum*self.dw[i]+eta*tsta1[-1].sum()*tsta[-1]/len(tsta1[-1])
                else :
                    tsta[i+1]=self.activation_deriv[i](tsta[i+1])#tsta[i+1]*(cellrange-tsta[i+1])       #每个cell输出量
                    
                    if i < len(self.weight)-2  :
                        tsta[i+2]=tsta[i+2][1 :]

                    tmp=self.weight[i+1]
                    tmp=(tmp.T*tsta[i+2]).T
                    #temp : w1ij * d1j
                    
                    tsta[i+1]=(tsta[i+1]*tmp.sum(axis=0))
                    ta=tsta[i+1][0]
                    for sb in range(1,len(tsta[i+1])) :
                        ta=tsta[i+1][sb]+ta
                    tsta[i+1]=ta

                    #d w c->d=eta*o(c)*d(d)+mom* d w c->d
                    tp=[]
                    for scy in range(1,len(tsta[i+1])) :
                     
                        tp.append(tsta[i].sum(axis=0)/len(tsta[i])*tsta[i+1][scy])

                    tp=np.array(tp)
                    self.dw[i]=momentum*self.dw[i]+eta*tp
            self.weight=self.weight+learning_rate*self.dw
                        

    def predict(self, X) :
        X = np.array(X)
        tsta=self.state.copy()
        tsta[0]=X.copy()
        #fw
        for i in range(len(self.weight)) :
            tsta[i]=np.insert(tsta[i], 0, values=np.ones(len(tsta[i])), axis=1)     #[1,index]
            tsta[i+1]=self.activation[i]( tsta[i].dot(self.weight[i].T) )*self.outrange
        res=tsta[-1]
        return res
    
    def binary_class(self,y) :
        y=np.array(y)
        f=y.mean()
        for i in range(len(y)) :
            if y[i]> f :
                y[i]=1
            else :
                y[i]=0
        return y

    def eve(self,pre,tuh) :
        ttt = 0
        ttf = 0
        tft = 0
        tff = 0
        for i in range(len(pre)) :
            if tuh[i]==1 :
                if tuh[i]==pre[i] :
                    ttt=ttt+1
                else :
                    ttf=ttf+1
            else :
                if tuh[i]==pre[i] :
                    tft=tft+1
                else :
                    tff=tff+1
        self.p=ttt/(ttt+ttf)
        self.r=ttt/(ttt+tff)#tp+fn
        
        return ttt*100/(ttt+ttf),tft*100/(tft+tff)
    
    def performence(self) :
        print("precision recall F1-score")
        print("precision : ",self.p)
        print("recall :    ",self.r)
        print("f1-score :  ",2/((1/self.p)+(1/self.r)))
        print()



# test
"""
nn = NN([2,3, 1])

aa = [[1, 0],[0,1],[0,0],[1,1]]
dd = [[1],[1],[0],[0]]


nn.fit(aa, dd, learning_rate=0.002,epochs=2000,momentum=0.25)


cc = nn.predict(aa)
print(cc)
"""
"""

y=np.array([[1,2,3],[4,5,6]])
x=np.array([[[1,2,1],[3,2,1],[2,1,2]],[[1,2,1],[3,2,1],[2,1,2]]])

temp=[]

for i in range(len(y)) :
    tmp=[]
    for j in range(len(y[i])) :
        tmp.append(x[i][j]*y[i][j])
    temp.append(np.array(tmp))"""
"""
a=np.array([[1,2,3],[4,5,6,7]])
a[0]=np.array(a[0])
a[1]=np.array(a[1])
b=np.array([[1,2],[2,3]])

print(a*3+a)
"""
