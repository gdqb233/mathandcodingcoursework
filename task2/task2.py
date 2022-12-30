import numpy as np
import torchvision
import pandas as pd
import random

#maybe needed for result analysis
#from sklearn.metrics import accuracy_score                 
#from sklearn.metrics import confusion_matrix        
#from sklearn.metrics import ConfusionMatrixDisplay


#np.random.seed(233)
#random.seed(233)


train_data = torchvision.datasets.MNIST(root='data/', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='data/', train=False)

train_data_data = train_data.data.numpy()
train_data_targets = train_data.targets.numpy()
test_data_data = test_data.data.numpy()
test_data_targets = test_data.targets.numpy()

def one_hot(X):                      
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

x_train, x_test = train_data_data / 255.0, test_data_data / 255.0 #

x_train = x_train.reshape(-1,784)  # flatten, (60000,28,28)（60000,784）
x_test = x_test.reshape(-1,784)  # flatten, (10000,28,28)（10000,784）

y_train = one_hot(train_data_targets) 
y_test = one_hot(test_data_targets) #

#----------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    x = np.where(x>=0,1,0)
    return x

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

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
        
        w_origin = self.w.copy()
        b_origin = self.b.copy() 
        
        for i in range(self.layernumber):       #in this example here i is 0. 1, 2. reshape w
           
            ww = self.w[i]                    
            if i == 0:                                   #the first dimension of the first layer is not dropped (784)
                ww = ww[:, self.indicelist[i+1]]            
            elif i == self.layernumber-1:              #other layers both first and second dimension is dropped
                ww = ww[self.indicelist[i]]            
            elif i != self.layernumber-1:                       # the last dimension of the last layer is not dropped(10)
                ww = ww[self.indicelist[i]][:, self.indicelist[i+1]]
            self.w[i] = ww
        
        for i in range(self.layernumber):            #reshape bs
            
            bb = self.b[i]
            if i != self.layernumber - 1:
                bb = bb[self.indicelist[i+1]]
            self.b[i] = bb
        
        return w_origin, b_origin

   
    def weight_matrices_reset(self, w_origin, b_origin):
        
        for i in range(self.layernumber):

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
        
     
    def fit(self, x, y, epoches = 10000, initial_lr = 0.05):
        
        #--------------------------------------------------------------------------------------initialising
        
        self.a.append(x)               #the first thing in a list should be x itself
        
        self.nodesnumberlist.insert(0, x.shape[1])        #insert the nodes of the input layer(784 here)        

        for i in range(self.layernumber):             #initialise dimensions here，no training。no dropout.
            
            ww = np.random.randn(self.nodesnumberlist[i], self.nodesnumberlist[i+1]) * 0.01
            self.w.append(ww.copy())                                 #get w list
        
            bb = np.random.randn(self.nodesnumberlist[i+1]) * 0.01
            self.b.append(bb.copy())
        

        #-----------------------------------------------------------------------------------------trainning
        
        for epoch in range(epoches):
            
            lr = initial_lr / (np.sqrt(epoch)+1)             ###################variable learning rate
                    
            self.a = []
            self.z = []
            self.a.append(x)
            
            #-------------------------------------------------------------------------------forward process
            
            ###########get the indice list first. Also not needed if dont do dropout
            self.indicelist = []
            for i in range(self.layernumber):
                activenodesnumber = int(self.nodesnumberlist[i] * self.activenodesratelist[i])
                activenodesindice = sorted(random.sample(range(0, self.nodesnumberlist[i]), 
                                      activenodesnumber))
                self.indicelist.append(activenodesindice.copy())
            
            ###########################dropout. Comment next sentence if do not need dropout
            w_origin, b_origin = self.dropout_weight_matrices()    
            
            for i in range(self.layernumber):
                #print(i)
                zz = np.dot(self.a[i], self.w[i]) + self.b[i]
                #self.z[i] = zz.copy()                    #calculate z
                self.z.append(zz.copy())
                
                if self.activationfunctionlist[i] == 'relu':
                    aa = relu(zz).copy()
                elif self.activationfunctionlist[i] == 'sigmoid':
                    aa = sigmoid(zz).copy()
                elif self.activationfunctionlist[i] == 'softmax':
                    aa = softmax(zz).copy()          
                #self.a[i+1] = aa.copy()                    #calculate a, notice here a[0] is x_train, so after that is a[i+1]
                self.a.append(aa.copy())
            
            y_temp = self.a[-1].copy()                 #temporary predict result
            
            

            y_test_temp = self.predict(x_test)

            if epoch % 100 == 0:                          #print something
                print('epoch', epoch, 'lr', lr, 'test accuracy:', accuracy(y_test_temp, y_test))
                print('crossentropyerror:', cross_entropy_error(y_temp, y), 'mse:', mean_squared_error(y_temp, y))

            #------------------------------------------------------------------------------backward process
            
            self.dw = []     #clear these four again
            self.db = []
            self.dz = []
            self.da = []
            
            for i in range(self.layernumber):
            
                if self.activationfunctionlist[self.layernumber-i-1] == 'softmax':            #dz
                    dzz = ( y_temp - y ) / x.shape[0]
                elif self.activationfunctionlist[self.layernumber-i-1] == 'relu':
                    dzz = relu_grad(self.a[self.layernumber-i]) * daa #/ x.shape[0]
                elif self.activationfunctionlist[self.layernumber-i-1] == 'sigmoid':
                    dzz = sigmoid_grad(self.a[self.layernumber-i]) * daa #/ x.shape[0]
                self.dz.insert(0, dzz.copy())
                
                dww = np.dot(self.a[self.layernumber-i-1].T, dzz)                            #get dw, needed
                self.dw.insert(0, dww.copy())
                
             
                daa = np.dot(dzz, self.w[self.layernumber-i-1].T)                                                   # da
                self.da.insert(0, daa.copy())
                
                dbb = np.sum(dzz, axis=0)                                                            #get db, needed
                self.db.insert(0, dbb)
            
            for i in range(self.layernumber):
                self.w[i] = (self.w[i] - lr * self.dw[i]).copy()
                self.b[i] = (self.b[i] - lr * self.db[i]).copy()

            ###############################dropout reset size. comment the next sentence if do not need dropout
            self.weight_matrices_reset(w_origin, b_origin)           #reset w, after backward propagation
        

        
    def predict(self, x):
        
        self.a2 = []
        self.z2 = []
            
        self.a2.append(x)
        for i in range(self.layernumber):
            
            zz = np.dot(self.a2[i], self.w[i]) + self.b[i]
            #self.z[i] = zz.copy()                    #calculate z
            
            if self.activationfunctionlist[i] == 'relu':
                aa = relu(zz)
            elif self.activationfunctionlist[i] == 'sigmoid':
                aa = sigmoid(zz)
            elif self.activationfunctionlist[i] == 'softmax':
                aa = softmax(zz)            
            #self.a[i+1] = aa.copy()                    #calculate a, notice here a[0] is x_train, so after that is a[i+1]
            self.a2.append(aa)
        
        y_temp = self.a2[-1]

        return y_temp
        


'''
#test accuracy 0.9102, loss still drecreasing after 10000 epoch, seems performance still improvable with more epoches
nn1  = NeuralNetwork()
nn1.add(DenseLayer(200, 'sigmoid', 0))
nn1.add(DenseLayer(100, 'relu', 0))
nn1.add(DenseLayer(10, 'softmax', 0))
nn1.fit(x_train, y_train, 10000, initial_lr = 2)
'''


#test accuracy 0.5658, maybe should try smaller learning rate?
nn2  = NeuralNetwork()
nn2.add(DenseLayer(200, 'sigmoid', 0.1))
nn2.add(DenseLayer(100, 'relu', 0.1))
nn2.add(DenseLayer(10, 'softmax', 0.1))
nn2.fit(x_train, y_train, 10000, initial_lr=1)


#y_pred = nn1.predict(x_test)
