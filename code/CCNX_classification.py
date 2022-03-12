# Code for standard X-rays classification

import numpy as np 
import pandas as pd
import os
from torchsummary import summary
import PIL
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import sys
import torch
from time import time
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms

# pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

##Please copy and paste the path of the directory with dataset
BASE_PATH='/home/bharat/Desktop/STAT_946_Data_Challenge_1/..../data'
train_dataset=pd.read_csv(os.path.join(BASE_PATH,'train_labels.csv'))
test_dataset=pd.read_csv(os.path.join(BASE_PATH,'test_labels.csv'))
print(train_dataset.shape())

#Plot Images to take a look at the Chest X-Ray images
fig=plt.figure(figsize=(32, 32)) #Size of Figure
columns = 3 #Number of Columns in the figure
rows = 5 #Number of Rows in the figure
for i in range(1,rows*columns+1):
    IMG_PATH=BASE_PATH+'train/'
    img=Image.open(os.path.join(IMG_PATH,train_dataset.iloc[i][0]))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

class Dataset(data.Dataset):
    def __init__(self,csv_path,images_path,transform=None):
        self.train_set=pd.read_csv(csv_path) #Read The CSV and then create the dataframe
        self.train_path=images_path #Path where images are present
        self.transform=transform # The transform function for augmenting images
    def __len__(self):
        return len(self.train_set)
    
    def __getitem__(self,idx):
        file_name=self.train_set.iloc[idx][0] 
        label=self.train_set.iloc[idx][1]
        img=Image.open(os.path.join(self.train_path,file_name)) #Loading Image
        if self.transform is not None:
            img=self.transform(img)
        return img,label

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
	transforms.ToTensor()])

training_set_untransformed=Dataset(os.path.join(BASE_PATH,'train_labels.csv'),os.path.join(BASE_PATH,'train/'))

new_created_images=[]
for j in range (len(training_set_untransformed)):
    if training_set_untransformed[j][1]==1:
        for k in range(8):
            transformed_image = transform_train(training_set_untransformed[j][0])
            new_created_images.append((transformed_image,1))
    else:
        transformed_image = transform_train(training_set_untransformed[j][0])
        new_created_images.append((transformed_image,0))

print(len(new_created_images))

train_size = int(0.8 * len(new_created_images))
validation_size = len(new_created_images) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(new_created_images, [train_size,validation_size])
training_generator = data.DataLoader(train_dataset,shuffle=True,batch_size=32,pin_memory=True)

if torch.cuda.is_available():
    device= torch.device('cuda:0')
    print('use of the GPU')
else:
    device= torch.device('cpu')

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
model.to(device)
print(summary(model, input_size=(3, 224, 224)))

PATH_SAVE='./Weights/'
if(not os.path.exists(PATH_SAVE)):
    os.mkdir(PATH_SAVE)

# PARAMETERS
learning_rate=1e-4
criterion = nn.CrossEntropyLoss()
lr_decay=0.99
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
history_accuracy=[]
history_loss=[]
epochs = 11


for epoch in range(epochs):  
    running_loss = 0.0
    correct=0
    total=0
    class_correct = [] #list(0. for _ in classes)
    class_total = [] #list(0. for _ in classes)
    
    for i, data in enumerate(training_generator, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        labels = eye[labels]
        optimizer.zero_grad()
        #torch.cuda.empty_cache()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)

        c = (predicted == labels.data).squeeze()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = float(correct) / float(total)
        
        history_accuracy.append(accuracy)
        history_loss.append(loss)
        
        loss.backward()
        optimizer.step()
        
        for j in range(labels.size(0)):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        
        running_loss += loss.item()
        
        print( "Epoch : ",epoch+1," Batch : ", i+1," Loss :  ",running_loss/(i+1)," Accuracy_train : ",accuracy)
    
    for k in range(len(classes)):
        if(class_total[k]!=0):
            print('Accuracy_training of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))
            print('[%d epoch] Accuracy of the network on the Training images: %d %%' % (epoch+1, 100 * correct / total))
        
torch.save(model.state_dict(), os.path.join(PATH_SAVE,'Last_epoch'+str(accuracy)+'.pth'))


plt.plot(history_accuracy)
plt.plot(history_loss)


test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
    transforms.ToTensor(),
    ])

correct_counter=0
for i in range(len(validation_dataset)): # già trasformate
    image_tensor = validation_dataset[i][0].unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    if index == validation_dataset[i][1]:
        correct_counter+=1
print("Accuracy=",correct_counter/len(validation_dataset))


'''
submission=pd.read_csv(BASE_PATH+'sample_submission.csv')
IMG_TEST_PATH=os.path.join(BASE_PATH,'test/')
# model load

def predict_image(image):
    image_tensor = test_transforms(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

for i in range(len(submission)):
    img=Image.open(IMG_TEST_PATH+submission.iloc[i][0])
    prediction=predict_image(img)
    submission_csv=submission_csv.append({'File_name': submission.iloc[i][0],'Label': prediction},ignore_index=True)
    if(i%10==0 or i==len(submission)-1):
        print('[',32*'=','>] ',round((i+1)*100/len(submission),2),' % Complete')
        
submission_csv.to_csv('submission_file.csv',index=False)
'''