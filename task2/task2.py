import numpy as np
import torchvision
import pandas as pd
import random

#import data with torchvision

train_data = torchvision.datasets.MNIST(root='data/', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='data/', train=False)

train_data_data = train_data.data.numpy()
train_data_targets = train_data.targets.numpy()
test_data_data = test_data.data.numpy()
test_data_targets = test_data.targets.numpy()

def onehot(X):                      
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

x_train, x_test = train_data_data / 255.0, test_data_data / 255.0 #

x_train = x_train.reshape(-1,784)  # flatten, (60000,28,28)（60000,784）
x_test = x_test.reshape(-1,784)  # flatten, (10000,28,28)（10000,784）

y_train = onehot(train_data_targets)   #onehot
y_test = onehot(test_data_targets) #onehot





def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    pass

def softmax(x):
	pass

	
def accuracy(y1, y2):

    y1 = np.argmax(y1, axis=1)
    y2 = np.argmax(y2, axis=1)

    accuracy = np.sum(y1 == y2) / float(y1.shape[0])
    return accuracy




#---------------------------------------------------------------------------

class DenseLayer:
    def __init__(self, nodes, activationfunction = 'relu', dropoutrate = 0.1):
        self.nodes = nodes
        self.activationfunction = activationfunction
        self.dropoutrate = dropoutrate
    
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------

class NeuralNetwork:
    def __init__(self):


        self.layernumber = 0

        self.nodesnumberlist = []     
        self.activationfunctionlist = []
        
        self.dropoutratelist = []
        self.activenodesratelist = []
    


        self.indicelist = []
        
        self.w = []
        self.b = []
        self.z = []
        self.a = []
        
        self.dw = []
        self.db = []
        self.dz = []
        self.da = []

		
    def dropout_weight_matrices(self):
              
        self.dropoutratelist[-1] = 0             #the last layer cannot do dropout
        
        w_origin = self.w.copy()
        b_origin = self.b.copy()
        #nodesnumberlist_origin = self.nodesnumberlist.copy()     
        
        for i in range(self.layernumber):       #in this example here i is 0. 1, 2
            #print(i)
            #print('在函数里面刚复制', len(w_origin[1]))
            
            ww = self.w[i]                    
            if i == 0:                                   #the first dimension of the first layer is not dropped (784)
                ww = ww[:, self.indicelist[i+1]]            
            elif i == self.layernumber-1:              #other layers both first and second dimension is dropped
                ww = ww[self.indicelist[i]]            
            elif i != self.layernumber-1:                       # the last dimension of the last layer is not dropped(10)
                ww = ww[self.indicelist[i]][:, self.indicelist[i+1]]
            self.w[i] = ww
            #dropout of the first and last layer and other layers are different.
            #print('切割操作完了', len(w_origin[1]))
        
        for i in range(self.layernumber):
            
            bb = self.b[i]
            if i != self.layernumber - 1:
                bb = bb[self.indicelist[i+1]]
            self.b[i] = bb
        
        return w_origin, b_origin


    
    def weight_matrices_reset(self, w_origin, b_origin):
        
        for i in range(self.layernumber):
            #print(i)
            if i == 0:
                w_origin[i][:, self.indicelist[i+1]] = self.w[i]
                self.w[i] = w_origin[i].copy()
            elif i == self.layernumber-1:   
                w_origin[i][self.indicelist[i]] = self.w[i]
                self.w[i] = w_origin[i].copy()
            elif i != self.layernumber-1:   
                temp = w_origin[i].copy()[self.indicelist[i]]
                temp[:, self.indicelist[i+1]] = self.w[i]
                w_origin[i][self.indicelist[i]] = temp
                self.w[i] = w_origin[i].copy()

        for i in range(self.layernumber):
            if i != self.layernumber-1:
                b_origin[i][self.indicelist[i+1]] = self.b[i]
                self.b[i] = b_origin[i].copy()

        
    def add(self, layer):
        self.nodesnumberlist.append(layer.nodes)
        self.layernumber += 1
        self.activationfunctionlist.append(layer.activationfunction)
        self.dropoutratelist.append(layer.dropoutrate)
        self.activenodesratelist.append(1-layer.dropoutrate)
		
	def fit(self, x, y, epoches = 3000, lr = 0.05):
		pass

	def predict(self, x):
		pass
		
		
#----------------------------------------------------------------------------------------
		
nn1  = NeuralNetwork()
nn1.add(DenseLayer(200, 'sigmoid'))
nn1.add(DenseLayer(100, 'relu'))
nn1.add(DenseLayer(10, 'softmax'))
nn1.fit(x_train, y_train)

