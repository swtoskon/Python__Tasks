
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 10 
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.01

# MNIST dataset 

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

print(len(train_dataset))
examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

# Fully connected neural network with one hidden layer
class Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784,10)
        nn.init.normal_(self.linear1.parameters,mean=0,std=0.1)
        nn.init.normal_(self.linear1.bias,mean=0,std=0.1)
        #self.linear2 = nn.Linear(128,68)
        self.linear2 = nn.Linear(10,10)
        nn.init.normal_(self.linear2.weight,mean=0,std=0.1)
        nn.init.normal_(self.linear2.bias,mean=0,std=0.1)
        
    def forward(self,X):
        X = F.relu(self.linear1(X))
        #X = F.relu(self.linear2(X))
        return F.log_softmax(self.linear2(X),dim=1)

model = Model(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)  

# Train the model
for epoch in range(10):
    model.train()
    running_loss =0
    for i, (images, labels) in enumerate(train_loader):  
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        #foward pass  + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(train_loader)))
          
    
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
correct_count, all_count = 0, 0
for images,labels in test_loader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))