
import shutil
import pandas as pd

sourcepath = r"D:\Projects\AllCelebImages\img_align_celeba\img_align_celeba"
destpath = r"D:\Projects\AllCelebImages\AttractivenessG"
attributes = pd.read_csv(r'C:\Users\Otto\Documents\1.Projects\data\list_attr_celeba.csv')
image_id = attributes.iloc[:]["image_id"]
attribute = attributes.iloc[:]["Attractive"]
gender = attributes.iloc[:]["Male"]


for i, name  in enumerate(image_id):
    path = r""
    if gender[i] == 1:
        path+="\\Male\\"
    else:
        path+="\\Female\\"
    if attribute[i] == 1:
        path+="\\Attractive\\"

    else:
        path+="\\NotAttractive\\"
         
    
    shutil.copyfile(sourcepath+r"\{}".format(name), 
                    destpath + path + name)
