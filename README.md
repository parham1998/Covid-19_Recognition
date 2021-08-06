# Covid-19_Recognition
Implementation of a pre-trained CNN for recognizing the covid-19 from chest X-Ray images with PyTorch library

### coronaviruses
![COVID-19](https://user-images.githubusercontent.com/85555218/127903018-7cd2ee42-7e15-4988-88ee-7ad6addfc347.png)

      Coronaviruses are a large family of viruses that were discovered in the 1960s. 
      these diseases are naturally present in birds and mammals. 
      But so far, seven types of coronaviruses have been discovered that can be transmitted from person to person. 
      One of these coronaviruses that have recently affected the lives of the world is an acute respiratory syndrome or Covid-19. 
      The outbreak of the virus started in December 2019 in Wuhan, China, and within a few months, it became a global epidemic.

### chest dataset
![chest_dataset](https://user-images.githubusercontent.com/85555218/127903321-afd46702-a945-4d69-af17-01596cf4ff6d.png)

      One way to diagnose the disease is to have a chest scan. That is, the chest is photographed and the health of the lungs is carefully monitored.
      Naturally, we need data to diagnose corona with deep learning. 
      
      The corona-positive CT scan data set that we will use for this training was collected by Joseph Cohen, Ph.D., University of Montreal.
      For corona-negative data or normal samples, we use the Chest X-Ray Images database from the Kaggle site. 

      Eventually, we will have a database in which we can train a deep network.

### code explanation

#### Prepare data: 
At this part, I downloaded the dataset from kaggle site (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and Joseph Cohen's GitHub (https://github.com/ieee8023/covid-chestxray-dataset) and separated them into train and test parts. I also made train.csv and test.csv to record images names and their corresponding labels. (if you don't want to change your train or test images, you can comment on this part)

#### Load data:
Here, I wrote a ChestDataset class to load the images in the tensor format and used DataLoader to separate images into batches.

#### one batch of images
![sample](https://user-images.githubusercontent.com/85555218/128534833-e265ad28-a717-4fbf-9a37-50122974611e.png)

#### CNN model:
I used the pre-trained VGG-16 network and changed the classifier part of the model for the binary classification problem. then trained the model on the aforementioned dataset.

#### vgg-16
![Screenshot (408)](https://user-images.githubusercontent.com/85555218/128536182-07f87459-d651-460b-98fe-1c4a894a572f.png)

#### hyperparameters:

loss function: binary cross entropy

optimizer: stochastic gradient descent

parameter | #
------------ | -------------
epoch  | 50
learning rate | 0.01
momentum  | 0.9
weight_decay | 0.0005
batch | 16
