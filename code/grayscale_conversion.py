# file per stabilire il tipo di grayscale per ogni immagine gray_scale

import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch

data_path= r'TrainSet\trainClinData.xls'
img_path= r'TrainSet\TrainSet'

df= pd.read_excel(data_path)
df.head()

tran= transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor()
])

# polmoni bianchi --> 0
# polmoni neri --> 1

label= [0, 0, 1, 1, 0, 0]
list_image= []

#Plot Images to take a look at the Chest X-Ray images
fig=plt.figure(figsize=(16, 16)) #Size of Figure
columns = 3
rows = 2
for i in range(1,rows*columns+1):
    img=Image.open(os.path.join(img_path,df.iloc[i][1]))
    img= tran(img).float()
    print(img.shape)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img.squeeze(), cmap='gray')
    list_image.append(img)
plt.show()

values= []
for i in range(len(list_image)):
    values.append(list_image[i].mean() - list_image[i].min())
print(values)

# comparare values vs. label

img_new= []
for i in range(10):
    img_new.append(255 - img[i])

# Invert gray_scale
plt.imshow(img.squeeze(), cmap='gray')

img.mean() - img.min()
img_new= 255 - img

plt.imshow(img_new.squeeze(), cmap='gray')
