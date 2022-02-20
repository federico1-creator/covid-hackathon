# basic setup for the network

import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torchvision.models.resnet import model_urls
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.misc
from PIL import Image
import json

print('Code to visualize the feature map')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image_start = Image.open(str('example\s1.jpg'))
#plt.imshow(image)
image_start.show() #plt

if torch.cuda.is_available():
    device= torch.device('cuda:0')
else:
    device= torch.device('cpu')

#model_urls['resnet101'] = model_urls['resnet101'].replace('https://', 'http://')
model = models.resnet18(pretrained=False)
model.to(device)
#print(model)

model_weights =[] # save the conv layer weights
conv_layers = [] # save name of conv layers
name_layers= []
model_children = list(model.children()) # to iterate on it
counter = 0

print(model_children[0])

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
        name_layers.append('conv n ' + str(counter))
        counter+=1
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    model_weights.append(child.weight)
                    conv_layers.append(child)
                    name_layers.append('conv n ' + str(counter))
                    counter+=1
print(f"Total convolution layers: {counter}")
print(name_layers)

image = transform(image_start) # 3, 224, 224
image = image.unsqueeze(0) # 1, 3, 224, 224
image = image.to(device)

outputs = []
names = []
for layer in conv_layers[0:]: #17
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))

print(outputs[0].shape)

processed = []
for feature_map in outputs: #17
    feature_map = feature_map.squeeze(0) # x, x, x
    gray_scale = torch.sum(feature_map, 0) # collpase all the channels
    gray_scale = gray_scale / feature_map.shape[0] # divided by n° channels
    processed.append(gray_scale.data.cpu().numpy())
print(processed[0].shape)

name_layers.append(len(name_layers))
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1) # 5*4
    plt.imshow(processed[i], cmap= 'gray')
    a.axis("off")
    a.set_title(name_layers[i], fontsize=30)

plt.savefig('feature_maps.jpg')
plt.show()

sns.heatmap(processed[0], xticklabels=False, yticklabels=False)
plt.show()