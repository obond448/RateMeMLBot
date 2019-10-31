import os
import torch
import torch.nn as nn
import pandas as pd #reading csv
import torchvision.transforms as transforms
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

def imshow(img):
    img = (img *(255/2)) + (255/2)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class MyNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, attr = sample['image'], sample['attr']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        #image.shape == [3, 281, 178]
        mean = torch.ones([218, 178]) * (255/2)
        image[0] = (image[0] - mean) / (255/2)
        image[1] = (image[1] - mean) / (255/2)
        image[2] = (image[2] - mean) / (255/2)
        
        return {'image': image,
                'attr': attr}
        
    
class MyTransform(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, attr = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        if (attr == -1):
            attr = 0
        
        return {'image': torch.from_numpy(image).float()    ,
                'attr': torch.tensor(attr, dtype=torch.float).long()}


class CelebrityDataset(Dataset):
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
        label = self.attributes.iloc[idx, 3]

        
        out = (image,label)

        if self.transform:
            out = self.transform(out)

        return (out['image'], out['attr']) #return image and attribute seperately
    
   
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
        x = self.pool(nn.functional.leaky_relu(self.conv1(x)))
        x = self.pool(nn.functional.leaky_relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 18*51*41) #reshape the tensor from 5*5 depth to 1
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    

trans = transforms.Compose([MyTransform(),
                            MyNormalize()])
dataset = CelebrityDataset(start = 0, end = 1000,csv_file='data/list_attr_celeba.csv',
                                    root_dir='data/img_align_celeba/img_align_celeba',
                                    transform=trans)

dataloader =  DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=0)




net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)


running_loss = 0
for i, (images, labels) in enumerate(dataloader):
    
    #zero all gradients
    optimizer.zero_grad()
    
    y_pred = net(images)
    
    loss = criterion(y_pred, labels)
    
    #backprop
    loss.backward()
    optimizer.step()
    
    
    # print statistics
    running_loss += loss.item()
    if i % 100 == 99:    # print every 2000 mini-batches
        print('[%5d] loss: %.3f' %
              (i + 1, running_loss / 100))
        running_loss = 0.0
        
facedataset2 = CelebrityDataset(start = 1000, end = 1005,csv_file='data/list_attr_celeba.csv',
                                    root_dir='data/img_align_celeba/img_align_celeba',
                                    transform=trans)
testloader = DataLoader(facedataset2, batch_size=1,
                        shuffle=False, num_workers=0)
with torch.no_grad():
    for (image, label) in testloader:
        
        y_pred = net(image)
        imshow(image.squeeze(0))
        print(y_pred, label)
    