import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
 
 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"       #environment variables



#0.9181
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)              
        x = x.view(-1, 64*4*4)
        x = self.fc(x)               
        x = nn.functional.normalize(x)
        return x

 
 
'''
#0.912
class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out 
'''


'''
#0.931
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,64,1,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)
        self.drop1 = nn.Dropout2d()
        self.fc6 = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        #print(" x shape ",x.size())
        x = x.view(-1,128*8*8)
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return x
'''


'''
#0.9267
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(1,40,2),
        nn.ReLU(),
        nn.MaxPool2d(2,1),
        nn.Conv2d(40,80,2),
        nn.ReLU(),
        nn.MaxPool2d(2,1),
        nn.Conv2d(80,160,3,padding = 1),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.MaxPool2d(3,3),)
        
        self.classifier = nn.Sequential(
        nn.Linear(160*8*8,200),
        nn.ReLU(),
#         nn.Linear(120,84),
#         nn.ReLU(),
#         nn.Linear(84,42),
#         nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(200,10))
        
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
'''      


device = "cuda:0" if torch.cuda.is_available() else "cpu"
 
batch_size = 256
num_workers = 0   
lr = 1e-4
epochs = 30
 
image_size = 28
 
data_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()])
 
train_data = datasets.FashionMNIST(root='./', train=True, download=True, transform=data_transform)
test_data = datasets.FashionMNIST(root='./', train=False, download=True, transform=data_transform)
 
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 

model = Net()      
#model = FashionCNN()  

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 

def train(epoch):
    model.train()                              
    train_loss = 0
    for data, label in train_loader:
        #data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch:', epoch, 'Training Loss:', train_loss)
 
def val(epoch):       
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            #data, label = data.cuda(), label.cuda()
            output = model(data)                       
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch:', epoch,  'Validation Loss:', val_loss, 'Accuracy:', acc)
 
if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        val(epoch)
 