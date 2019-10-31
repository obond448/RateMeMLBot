import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform
import numpy as np
import torch.nn as nn
import torch.optim as optim

def imshow(img):
    img = img /2 + .5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
#def load_dataset():
#    data_path = './data/img_align_celeba/img_align_celeba'
#    transform1 = transforms.Compose([
#            transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#    train_dataset = torchvision.datasets.ImageFolder(
#        root=data_path,
#        transform=transform1
#    )
#    train_loader = torch.utils.data.DataLoader(
#        train_dataset,
#        batch_size=64,
#        num_workers=0,
#        shuffle=True
#    )
#    return train_loader
    
class MyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, attr = sample['image'], sample['attr']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image).float()    ,
                'attr': torch.tensor(attr, dtype=torch.float).long()}
        
#class MyNormalize(object):
#    """Convert ndarrays in sample to Tensors."""
#
#    def __call__(self, sample):
#        image, attr = sample['image'], sample['attr']
#
#        # swap color axis because
#        # numpy image: H x W x C
#        # torch image: C X H X W
#        image = transforms.Normalize((.5,.5,.5),(.5,.5,.5))
#        return {'image': torch.from_numpy(image),
#                'attr': torch.tensor(attr, dtype=torch.uint8)}
        
class CelebDataset(Dataset):
    def __init__(self, start, end, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attributes = pd.read_csv(csv_file)
        self.attributes = self.attributes.iloc[start:end,:]
        self.root_dir = root_dir
        self.transform = transform
        
        
    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.attributes.iloc[idx, 0])
        image = io.imread(img_name)
        attr = self.attributes.iloc[idx, 3]

        sample = {'image': image, 'attr': attr}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(9, 18, 5)
        #pool again here
        self.fc1 = nn.Linear(18*51*41, 80)
        self.fc2 = nn.Linear(80, 2)
    
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 18*51*41) #reshape the tensor from 5*5 depth to 1
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    
trans = transforms.Compose([
MyToTensor()])
  
facedataset = CelebDataset(start = 0, end = 1000,csv_file='data/list_attr_celeba.csv',
                                    root_dir='data/img_align_celeba/img_align_celeba',
                                    transform=trans)

#fig = plt.figure()
#s = facedataset[1]
#print(s['image'])
#imshow(s['image'])


#plt.show()

dataloader =  DataLoader(facedataset, batch_size=1,
                        shuffle=False, num_workers=0)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

epoch = 0
running_loss = 0
for i, data in enumerate(dataloader):
    inputs = data['image']
    #inputs = inputs.type('torch.DoubleTensor')
    label = data['attr']
    if label == torch.tensor([-1]):
        label = torch.tensor([0])
    else:
        label = torch.tensor([1])
        
    optimizer.zero_grad()
    
    y_pred = net(inputs)
    
    loss = criterion(y_pred, label) 
    
    
    
    loss.backward()
    optimizer.step()
    
     # print statistics
    running_loss += loss.item()
    if i % 100 == 99:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0


facedataset2 = CelebDataset(start = 1000, end = 1005,csv_file='data/list_attr_celeba.csv',
                                    root_dir='data/img_align_celeba/img_align_celeba',
                                    transform=trans)
testloader = DataLoader(facedataset2, batch_size=1,
                        shuffle=False, num_workers=0)
with torch.no_grad():
    for data in testloader:
        inputs = data['image']
        label = data['attr']
        if label == torch.tensor([-1]):
            label = torch.tensor([0])
        else:
            label = torch.tensor([1])
        
        y_pred = net(inputs)
        imshow(data['image'].squeeze(0))
        print(y_pred, label)
#fig = plt.figure()
#
#for i in range(len(face_dataset)):
#    
#    sample = face_dataset[i]
#
#    print(i, sample['image'].shape, sample['attr'])
#
#    ax = plt.subplot(1, 4, i + 1)
#    plt.tight_layout()
#    ax.set_title('Sample #{}'.format(i))
#    ax.axis('off')
#    plt.imshow(sample['image'])
#
#    if i == 3:
#        plt.show()
#        break
#
#attributes = pd.read_csv("data/list_attr_celeba.csv")
#print(len(attributes))