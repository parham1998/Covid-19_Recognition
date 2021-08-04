# =============================================================================
# Import required libraries
# =============================================================================
import os
import csv
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from PIL import Image
import timeit

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

# =============================================================================
# Check if CUDA is available
# =============================================================================
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

# =============================================================================
# Prepare data (run this part just 1 time if you wanna change your data else ignore this part) 
# =============================================================================
csvPath = './metadata.csv'
df = pd.read_csv(csvPath)

# from the metadata file, I chose the patients that have COVID-19 and add their filenames to a list
p_id = 0
cov = []
covid_plus = []
for (i, series) in df.iterrows():
    patient_id = series['patientid']
    if patient_id != p_id and len(cov) > 0 and p_id != 0:
        covid_plus.append(cov)
        cov = []
    if series["finding"] == "Pneumonia/Viral/COVID-19" and series["view"] == "PA":
        cov.append(series["filename"])
    p_id = series['patientid']

# split train and test data 
x_train_c, x_test_c = train_test_split(covid_plus, test_size=0.20, random_state=23)

# download covid chest x-ray images from JOSEPH PAUL COHEN github
for img in sum(x_train_c, []):
    src = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/' + img
    response = requests.get(src).content
    open('./dataset/train/covid/' + img, 'wb').write(response)
for img in sum(x_test_c, []):
    src = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/' + img
    response = requests.get(src).content
    open('./dataset/test/covid/' + img, 'wb').write(response)
    
# download normal chest x-ray images from Kaggle
'''
I downloaded the whole dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
and chose 201 images from kaggle/train/NORMAL/ randomly as split them into train and test.
'''
samples = 201
filenames = os.listdir('./kaggle/train/NORMAL/')
random.seed(42)
filenames = random.sample(filenames, samples)
for i in range(samples):
    img = cv2.imread('./kaggle/train/NORMAL/' + filenames[i])
    if i < 165:
        cv2.imwrite('./dataset/train/normal/' + filenames[i], img)
    else:
        cv2.imwrite('./dataset/test/normal/' + filenames[i], img)
#  
def toCSV(root, covid, normal):
    with open(root, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'label'])
        for img in covid:
            writer.writerow([img, '1'])
        for img in normal:
            writer.writerow([img, '0'])        

x_train_covid = sorted(os.listdir('./dataset/train/covid/'))
x_train_normal = sorted(os.listdir('./dataset/train/normal/'))
x_test_covid = sorted(os.listdir('./dataset/test/covid/'))
x_test_normal = sorted(os.listdir('./dataset/test/normal/'))

toCSV('./dataset/train.csv', x_train_covid, x_train_normal)
toCSV('./dataset/test.csv', x_test_covid, x_test_normal)

# now save all your train data in train file and save all your test data in test file

# =============================================================================
# Load data
# =============================================================================
class ChestDataset(torch.utils.data.Dataset):                                 
    def __init__(self, root, csv_file, transforms=None):                       
        self.root = root                   
        self.csv_file = shuffle(pd.read_csv(csv_file))
        self.transforms = transforms            
    
    def __getitem__(self, idx):                                                 
        img_name = os.path.join(self.root, self.csv_file.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(self.csv_file.iloc[idx, 1])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label                                                      
    
    def __len__(self):                                                          
        return len(self.csv_file)  
  

transform_data = transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize(
                          mean=[0.5, 0.5, 0.5], 
                          std=[0.5, 0.5, 0.5]
                      ),
                  ])    

trainset = ChestDataset(root='./dataset/train/', 
                        csv_file='./dataset/train.csv', 
                        transforms=transform_data)
testset = ChestDataset(root='./dataset/test/', 
                        csv_file='./dataset/test.csv', 
                        transforms=transform_data)

# show one image
def imshow(img):
    img = (img / 2) + 0.5
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

img, label = trainset[200]
imshow(img)

#
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=16,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=16,
                                         shuffle=False)

# =============================================================================
# Show one batch of images
# =============================================================================
classes = ['normal', 'covid']

# get one batch of images
images, labels = iter(trainloader).next()

# plot the images with corresponding labels
fig = plt.figure(figsize=(16, 16))
for i in np.arange(16):
    ax = fig.add_subplot(4, 4, i+1)
    imshow(images[i])
    ax.set_title(classes[labels[i]])
    
# =============================================================================
# CNN models
# =============================================================================
PATH ='./checkpoints/cifar10_VGG16_pretrained.pth'

net = torchvision.models.vgg16()

net.classifier = nn.Sequential(
    nn.Linear(in_features=512*7*7, out_features=64),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=64, out_features=1),
    nn.Sigmoid()
    )

print(net)    

for param in net.features.parameters():
    param.requires_grad = False

net.cuda()
    
# =============================================================================
# Load model
# =============================================================================
net.load_state_dict(torch.load(PATH))  
net.cuda()

# =============================================================================
# Specify loss function and optimizer
# =============================================================================
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
epochs = 50

criterion = nn.BCELoss()
params = [p for p in net.parameters() if p.requires_grad == True]
optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# =============================================================================
# training
# =============================================================================
best_accuracy = 0

# losses per epoch
train_losses = []
test_losses = []

# ===========
# train model
# ===========
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, targets) in enumerate(trainloader):
                
        if train_on_GPU:
            images, targets = images.cuda(), targets.cuda()

        # zero the gradients parameter
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = net(images).reshape(-1)
                
        # calculate the batch loss
        loss = criterion(outputs, targets.float())
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
    
        # parameters update
        optimizer.step()

        train_loss += loss.item()
        predicted = []
        for o in outputs:
            if o >= 0.5:
                predicted.append(1)
            else:
                predicted.append(0)
        predicted = torch.Tensor(predicted).cuda()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    train_losses.append(train_loss/(batch_idx+1))
    print('Epoch: {} \t Training Loss: {:.3f} \t Training Accuracy: {:.3f}'.format(epoch+1, train_loss/(batch_idx+1), 100.*correct/total))

# ==============
# test model
# ==============
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
        
            if train_on_GPU:
                images, targets = images.cuda(), targets.cuda()
            
            outputs = net(images).reshape(-1)
                        
            loss = criterion(outputs, targets.float())

            test_loss += loss.item()
            predicted = []
            for o in outputs:
                if o >= 0.5:
                    predicted.append(1)
                else:
                    predicted.append(0)
            predicted = torch.Tensor(predicted).cuda()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    acc = 100.*correct/total
    test_losses.append(test_loss/(batch_idx+1))
    print('Epoch: {} \t Test Loss: {:.3f} \t Test Accuracy: {:.3f}'.format(epoch+1, test_loss/(batch_idx+1), acc))
    
    # save model if test accuracy has increased 
    global best_accuracy
    if acc > best_accuracy:
        print('Test accuracy increased ({:.3f} --> {:.3f}). saving model ...'.format(best_accuracy, acc))
        torch.save(net.state_dict(), PATH)
        best_accuracy = acc

print('==> Start Training ...')
for epoch in range(epochs):
    start = timeit.default_timer()
    train(epoch)
    test(epoch)
    stop = timeit.default_timer()
    print('time: {:.3f}'.format(stop - start))
print('==> End of training ...')

# =============================================================================
# Test model on test data & Confusion matrix
# =============================================================================
test = torch.utils.data.DataLoader(testset,
                                          batch_size=90,
                                          shuffle=True)
images, labels = iter(test).next()
outputs = net(images.cuda()).reshape(-1)
predicted = []
for o in outputs:
    if o >= 0.5:
        predicted.append(1)
    else:
        predicted.append(0)
cm = confusion_matrix(labels.cpu().detach().numpy(), np.array(predicted))