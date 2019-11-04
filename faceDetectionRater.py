import cv2, sys
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def imshow(data_img):
    data_img[0] = (data_img[0] + 0.485) * .229 
    data_img[1] = (data_img[1] + .456) * .224
    data_img[2] = (data_img[2] + .406) * .225
    plt.imshow(np.transpose(data_img, (1,2,0)))
    

#file = r"C:\Users\Otto\Documents\1.Projects\Celebrity\imageME2.jpg"


def run(inputs):
    
    input_size = 224
    num_classes = 2
    model = models.squeezenet1_0(pretrained=False)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    
    #model = TheModelClass(*args, **kwargs)
    dict_load = torch.load(r"modelAttract.pt")
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
        # un comment for outputs
        # print(outputs)
        # print(likelyhoods)
        # imshow(inputs[0])
        
        return likelyhoods.numpy()[0]
    
    

def findFace(imagePath):
    # Get user supplied values
    imagePath = imagePath
    cascPath = "haarcascade_frontalface_default.xml"
    
    
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    # Read the image
    
    image = cv2.imread(imagePath)
            
    if image.shape == None: #failed read    
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #min size proportional
    minsize = (int(image.shape[0]/2), int(image.shape[1]/2))
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    
    # print("Found {0} faces!".format(len(faces)))
    
    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    
    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
    
    #return just the first face detected
    if (len(faces) > 0):
        imagePos = {}
        
        for i, d in enumerate(faces):
            
            (x,y,w,h) = faces[i]
            
            pad = int( ((w+h)/2) * 1/4)
            ymax = min(y+h+pad, image.shape[0]-1)
            xmax = min(x+w+pad, image.shape[1]-1)
            y = max(y - pad, 0)
            x = max(x - pad, 0)
            
            imagePos[i] = [y, ymax ,x, xmax]
            
        return imagePos  
    else:
        return None



def rate_faces_save(imgfile, name="None"):
    
    img = imgfile
    imgs = findFace(img)
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    conf = {}
    fig,ax = plt.subplots(1)
    
    if imgs != None:
        plt.imshow(np.array(image))
        for i in imgs.keys():
            #print("figure {}".format(i))
            y, ymax, x, xmax = imgs[i]
            conf[i] = run(Image.fromarray(image[y:ymax, x:xmax]))
            
            rect = patches.Rectangle((x,y),xmax-x,ymax-y,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            ax.text(x,y,"{num:.2f}".format(num=conf[i][0]), color="green")
        if name == "None":
            plt.show()
        else:
            fig.savefig(name, dpi=400)
            plt.close(fig)
            
    
    
if __name__ =="__main__": #this is to stop the main code running when this code is imported to another script
    img = "k1puewb6uyv31.jpg"
    rate_faces_save(img, "myimg.jpg")