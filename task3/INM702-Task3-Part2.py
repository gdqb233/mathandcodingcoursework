#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


# In[10]:


dataset_path = './natural-images/data/natural_images'
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
    
print(torch.cuda.is_available())
devices = [d for d in range(torch.cuda.device_count())]
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(device_names)


# In[11]:


# The folder contains a subfolder for each class of shape
classes = sorted(os.listdir(dataset_path))
print(classes)


# In[12]:


from PIL import Image

# function to resize image
def resize_image(src_image, size, bg_color="white"): 
    from PIL import Image, ImageOps 
    
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)
    
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
  
    # return the resized image
    return new_image


# In[13]:


def load_dataset(data_path):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    # Load all the images
    transformation = transforms.Compose([transforms.ToTensor()])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    
    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    
    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    
    # define a loader for the training data we can iterate through in 50-image batches
    demo_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False
    )
        
    return train_loader, test_loader, demo_loader


# In[14]:


#  a method that return the name of class for the label number
def output_label(label):
    output_mapping = {
                 0: "airplane",
                 1: "car",
                 2: "cat",
                 3: "dog",
                 4: "flower", 
                 5: "fruit", 
                 6: "motorbike",
                 7: "person"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


# In[15]:


# New location for the resized images
dataset_resized = './natural-images/data/natural_images_resized'

# Create the output folder if it doesn't already exist
if os.path.exists(dataset_resized):
    shutil.rmtree(dataset_resized)
    
# Create resized copies of all of the source images
size = (64,64)

# Loop through each subfolder in the input folder
print('Resizing images...')
for root, folders, files in os.walk(dataset_path):
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        # Create a matching subfolder in the output dir
        saveFolder = os.path.join(dataset_resized,sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        # Loop through the files in the subfolder
        file_names = os.listdir(os.path.join(root,sub_folder))
        for file_name in file_names:
            # Open the file
            file_path = os.path.join(root,sub_folder, file_name)
            #print("reading " + file_path)
            image = Image.open(file_path)
            # Create a resized version and save it
            resized_image = resize_image(image, size)
            saveAs = os.path.join(saveFolder, file_name)
            #print("writing " + saveAs)
            resized_image.save(saveAs)

print('Done.')


# In[16]:


# Get the iterative dataloaders for test and training data
train_loader, test_loader, demo_loader = load_dataset(dataset_resized)
batch_size = train_loader.batch_size
print("Data loaders ready to read", dataset_resized)


# In[17]:


demo_batch = next(iter(demo_loader))
images, labels = demo_batch
print(type(images), type(labels))
print(images.shape, labels.shape)

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15, 20))
plt.imshow(np.transpose(grid, (1, 2, 0)))
print("labels: ", end=" ")
for i, label in enumerate(labels):
    print(output_label(label), end=", ")


# In[26]:


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=32*15*15, out_features=1000)
        #self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=1000, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=8)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


# In[27]:


device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"

# Create an instance of the model class and allocate it to the device
model = CNN()
model.to(device)


# In[28]:


error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)


# In[29]:


# Training a network and Testing it on test dataset


# In[30]:


num_epochs = 10
count = 0
# Lists for visualization of loss and accuracy 
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
        train = Variable(images.view(50, 3, 64, 64))
        labels = Variable(labels)
        
        # Forward pass 
        outputs = model(train)
        loss = error(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
        
        # Print metrics so we see some progress
        #print('\tTraining batch {} Loss: {:.6f}'.format(count + 1, loss.item()))
    
        count += 1
    
    # Testing the model
    
        if not (count % 10):    # It's same as "if count % 50 == 0"
            total = 0
            correct = 0
        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                labels_list.append(labels)
            
                test = Variable(images.view(50, 3, 64, 64))
            
                outputs = model(test)
            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
            
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data.detach().cpu().numpy())
            # loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy.detach().cpu().numpy())
            # accuracy_list.append(accuracy)
        
        if not (count % 96):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
    print(count)


# In[31]:


torch.save(model, "Part2cnn1.pth")


# In[32]:


plt.plot(iteration_list, loss_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Iterations vs Loss")
plt.show()


# In[33]:


plt.plot(iteration_list, accuracy_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy")
plt.show()


# In[34]:


from itertools import chain 
import sklearn.metrics as metrics

predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

confusion_matrix(labels_l, predictions_l)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))


# In[ ]:





# In[ ]:




