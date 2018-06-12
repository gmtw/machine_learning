
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

class Perceptron(object):
    #Perceptron Classifier
    #Parameters
    #----------
    #eta : float
    # Learning rate (between 0.0 and 1.0)
    #n_iter: int
    #Pasos sobre conjunto de entrenamiento
    #Atributos
    #-----------
    #w_ :arreglo 1D
    #errors_ : lista
    # numero de clasificaciones erroneas en cada lapso de tiempo
    def __init__(self, eta=0.01, n_iter=10):
        self.eta=eta
        self.n_iter= n_iter
    def fit(self, X, y):
        #Ajuste de entrenamiento de datos
        
        #Parametros
        #-----------
        #X:{arreglo}, shape={n_samples, n_ajustes}
        #vectores de entrenamiento, donde n_samples de muestras n_features es el número de ajustes
        #y:{arreglo}, shape={n_samples}
        #Valores objetivo
        #Returns
        #-------
        #self:object
        
        self.w_=np.zeros(1+X.shape[1])
        self.errors_ =[]
        
        for _ in range(self.n_inter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:]+= update*xi
                self.w[0] +=update
                errors += int(update !=0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self,X):
    #calcula entrada net
        return np.dot(X,self.w_[1:]+self.w_[0])
    
    def predict (self,X):
        #regresa la etiqueta class despues de un paso unitario
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    


# In[3]:

import pandas as pd


# In[4]:

#df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
df=pd.read_csv("/home/julio/Escritorio/iris.data",header=None)


# In[5]:

df.tail()


# In[6]:

import matplotlib.pyplot as plt


# In[7]:

y=df.iloc[0:100,4].values


# In[8]:

y=np.where(y=="Iris-setosa",-1,1)


# In[9]:

X=df.iloc[0:100,[0,2]].values


# In[10]:

plt.scatter(X[:50,0],X[:50,1],color="red",marker="o", label="setosa")


# In[11]:

plt.scatter(X[50:100,0],X[50:100,1],color="blue",marker="x", label="versicolor")


# In[12]:

plt.xlabel("petal lenght")
plt.ylabel("sepal lenght")
plt.legend(loc="upper left")
plt.show()


# In[13]:

ppn= Perceptron(eta=0.1,n_iter=10)


# In[14]:

ppn.fit(X,y)


# In[ ]:



