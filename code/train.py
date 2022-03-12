import torch
import torch.optim as optim
import torchvision
from torchsummary import summary
from torchvision.models.resnet import model_urls

torch.manual_seed(42)
# load the data
dataset= ''
#train, evaluation= torch.utils.data.random_split(dataset, [1000, 103])
loader= ''

# load the model
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#model_urls['resnet34'] = model_urls['resnet34'].replace('https://', 'http://')
model= torchvision.models.resnet34(pretrained=False)
num= model.fc.in_features
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc= torch.nn.Linear(512, 2)

summary(model, (3,224,224))

if torch.cuda.is_available():
    model.to('cuda')
    input_batch = input_batch.to('cuda')

# data loader
model.train()
loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1])) # for unbalanced class
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data #[inputs, labels]
        #inputs, labels = data[0].to(device), data[1].to(device) # per ogni step che carica dei dati, nella GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')

model_path = './model.pth'
torch.save(model.state_dict(), model_path)


# model load
model.load_state_dict(torch.load(model_path))

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for data in testloader:
        images, labels = data

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    '''output = model(input_batch)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
output[0]
probabilities[0]'''
print('accuracy: ', 100 * correct // total)