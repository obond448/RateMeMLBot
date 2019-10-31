import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(data_img):
    data_img[0] = (data_img[0] + 0.485) * .229 
    data_img[1] = (data_img[1] + .456) * .224
    data_img[2] = (data_img[2] + .406) * .225
    plt.imshow(np.transpose(data_img, (1,2,0)))
    

file = r"C:\Users\Otto\Documents\1.Projects\Celebrity\imageOTHER4.jpg"



def run(inputs):
    
    input_size = 224
    num_classes = 2
    model = models.squeezenet1_0(pretrained=False)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    
    #model = TheModelClass(*args, **kwargs)
    dict_load = torch.load(r"C:\Users\Otto\Documents\1.Projects\Celebrity\modelAttract.pt")
    model.load_state_dict(dict_load["model_state_dict"])
    model.eval()
    
    transform =  transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    #inputs = Image.open(file)
    inputs = transform(inputs)
    inputs = inputs.view(1,3,224,224)
    
    with torch.no_grad():
        outputs = model(inputs) #inputs is a array with batch_size elements
                 
        _, predicted = torch.max(outputs, 1)
        
        likelyhoods = torch.nn.functional.softmax(outputs, dim = 1)
        print(outputs)
        print(likelyhoods)
        imshow(inputs[0])
        

    
    