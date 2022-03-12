'''
File con le funzioni base utilizzate
'''
data_path= r'TrainSet/trainClinData.xls'

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

def calculate_mean_std(dataset):
    means=[]
    stds= []
    for img in subset(dataset):
        # load correctly the images
        means.append(torch.mean(img))
        stds.append(torch.std(img))
    mean= torch.mean(torch.tensor(means))
    std= torch.mean(torch.tensor(stds))
    return mean, std

transf_augmented= transforms.Compose([
    #transforms.ToPIL(),
    
    transforms.ToTensor()
])
transf_loader= transforms.Compose([
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
            label= self.dataframe['Prognosis'][index]

        else:
            file= open(self.metadata)
            data= json.load(file)
            change_path= data['image_name'][index]
            img_name= os.path.join(self.root_dir, change_path)
            label= data['label'][index]

        print(img_name)
        # apply function for the pre-processing
        img= Image.open(img_name)
        # data augmentation
        img= self.transform(img)
        return img, label 


#custom dataset ...
#data loader ...
imag= Image.open('TrainSet\TrainSet\P_1.png')
print(np.unique(imag))

print('use custom datatset')
dataset= custom_dataset(excel_file=data_path, root_dir='TrainSet\TrainSet', transform=transf_augmented, mode='augmented')

# create the augmented dataset and metadata for the training
img_path= []
label_img= []

i= len(dataset)*0 # to increase at different iteration 
for img, label in dataset:
    print(img.shape)
    print(np.unique(img))

    save_image(img.float(), 'augmented/img_%d.png' % i)
    img_path.append('img_%d.png' % i)
    label_img.append(label)
    i+=1

data= {'image_name': img_path, 'label': label_img}
with open('metadata_augmented.json', 'w') as file:
    json.dump(data, file)

# data part for the training
dataset_train= custom_dataset(metadata= 'metadata_augmented.json', root_dir='augmented', transform=transf_loader)
loader= DataLoader(dataset_train, batch_size=32, shuffle=True)

# training ...

# test/eval
dataset_test= custom_dataset(excel_file= '', root_dir='', transform=transf_loader)
loader_test= DataLoader(dataset_test, batch_size=32, shuffle=False)
