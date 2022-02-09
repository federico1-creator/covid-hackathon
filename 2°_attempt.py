'''
Complete code for the SIMPLE process of the images, in this file we are using the images without any changes.
1 --> 'SEVERE'
0 --> 'MILD'
'''
import os
import argparse
import json
from PIL import Image
from torchsummary import summary
from torchvision.models.resnet import model_urls
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import pandas as pd

torch.manual_seed(42)
data_path= r'TrainSet/trainClinData.xls'

print('Start with the processing of the data')
# TODO: we have to add the normalization and augmentation part

transf_loader= transforms.Compose([
    transforms.Resize([1000, 1000]),
    transforms.ToTensor()
])

class custom_dataset(Dataset):
    # load data
    def __init__(self, root_dir, transform=transf_loader, excel_file= None, metadata=None, mode='train_phase'):
        self.dataframe= pd.read_excel(excel_file)
        self.root_dir= root_dir
        self.transform= transform
        self.metadata= metadata
        self.mode= mode
    
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        if self.mode == 'augmented':
            change_path= self.dataframe['ImageFile'][index]
            img_name= os.path.join(self.root_dir, change_path)
            onehot_label= self.dataframe['Prognosis'].astype('category').cat.codes
            onehot_label_float= torch.tensor(onehot_label, dtype=torch.long)
            label= onehot_label_float[index]

        else:
            file= open(self.metadata)
            data= json.load(file)
            change_path= data['image_name'][index]
            img_name= os.path.join(self.root_dir, change_path)
            label= data['label'][index]

        #print(img_name)
        # apply function for the pre-processing
        img= Image.open(img_name)
        # data augmentation
        img= self.transform(img).float()
        return img, label 

dataset= custom_dataset(excel_file=data_path, root_dir='TrainSet\TrainSet', transform=transf_loader, mode='augmented')
print(len(dataset))
train, test= torch.utils.data.random_split(dataset, [1000, 103])
print('Load the data')

train_loader= DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
test_loader= DataLoader(test, batch_size=32, shuffle=False)

print('Load the model')
model_urls['resnet34'] = model_urls['resnet34'].replace('https://', 'http://')
model= torchvision.models.resnet34(pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 1 channel in input image
num= model.fc.in_features
model.fc= torch.nn.Linear(512, 1)

#summary(model, (1,224,224))

print('Optimizer')
if torch.cuda.is_available():
    model.to('cuda')
    print('GPU used')

model.train()
criterion= torch.nn.BCEWithLogitsLoss()
loss = torch.nn.CrossEntropyLoss() # weight=torch.tensor([1, 1])    for unbalanced class
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('Start the training ...')
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data #[inputs, labels]
        #inputs, labels = data[0].to(device), data[1].to(device) # per ogni step che carica dei dati, nella GPU
        optimizer.zero_grad() # FIXME: posizione corretta?
        outputs = model(inputs)
        labels= labels.reshape([32, 1]).float()
        loss = criterion(outputs, labels) # FIXME: sigmoid in the output ???
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(i)
        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
print('Finished Training')

model_path = 'saved_models/model_pretrain.pth'
torch.save(model.state_dict(), model_path)
print('Saved the model')

model.load_state_dict(torch.load(model_path))
print('Loaded the model')

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for data in test_loader:
        images, labels = data

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('accuracy: %d' %(100 * correct // total) + '%')
# 46% accuracy
