# packages
import numpy as np
import pandas as pd
import os
import cv2
import shutil as sh
import matplotlib.pyplot as plt
from .. import config

# path varibles and constant
input_dir=config.INPUT_DIR
base_dir=config.BASE_DIR
img_size = config.image_size
num_classes=config.num_classes
channel = 3

train_dir=config.TRAIN_DIR

val_dir=config.VAL_DIR
test_dir=config.TEST_DIR

def sorting(x):
    d,l=x.split("_")
    l=l.split(".")[0]
    d=int(d)
    s=.1 if l=="left" else .2
    return d+s




#################################################################
#######   DIRECTORY MAKING    ###################################
################################################################

#make a train directory & subfolders containing samples from each classes
##train_dir=os.path.join(base_dir,"train")
##os.makedirs(train_dir)
train_0_dir=os.path.join(train_dir,"0")
os.makedirs(train_0_dir,exist_ok=True)

train_1_dir=os.path.join(train_dir,"1")
os.makedirs(train_1_dir,exist_ok=True)

train_2_dir=os.path.join(train_dir,"2")
os.makedirs(train_2_dir,exist_ok=True)

train_3_dir=os.path.join(train_dir,"3")
os.makedirs(train_3_dir,exist_ok=True)

train_4_dir=os.path.join(train_dir,"4")
os.makedirs(train_4_dir,exist_ok=True)

#make a validation directory& subfolders containing samples from each classes
##val_dir=os.path.join(base_dir,"validation")
##os.makedirs(val_dir)
val_0_dir=os.path.join(val_dir,"0")
os.makedirs(val_0_dir,exist_ok=True)

val_1_dir=os.path.join(val_dir,"1")
os.makedirs(val_1_dir,exist_ok=True)

val_2_dir=os.path.join(val_dir,"2")
os.makedirs(val_2_dir,exist_ok=True)

val_3_dir=os.path.join(val_dir,"3")
os.makedirs(val_3_dir,exist_ok=True)

val_4_dir=os.path.join(val_dir,"4")
os.makedirs(val_4_dir,exist_ok=True)

#make a test directory

##test_dir=os.path.join(base_dir,"test")
##os.makedirs(test_dir)

#******************* END OF DIRECTORY MAKING************************



######################################################################
#   TRANSFERING RESPECTIVE IMAGES INTO FOLDERS
#
########################################################################

def placing_image(image_series,dest_train,dest_val):
    df=image_series
    length=image_series.size
    #shuffle
    shuffled=image_series.reindex(np.random.permutation(image_series.index))

    #now split it into train and validation set
    no_train=int(length*.8)
    
    train_indexes=df[0:no_train]
    val_indexes=df[no_train:length]

    #now transfer them into train and validation folder
    for image in train_indexes:
        input_dirs=os.path.join(input_dir,image+".jpeg")
        sh.move(input_dirs,dest_train)
        
    for image in val_indexes:
        input_dirs1=os.path.join(input_dir,image+".jpeg")
    
        sh.move(input_dirs1,dest_val)
    

        
# loading label
df = pd.read_csv(os.path.join(config.dataset_path(),"eyepack","trainLabels.csv"))

#first seperate the image name from trainlabellis.csv
class_0=df.loc[df["level"]==0,"image"]

class_1=df.loc[df["level"]==1,"image"]

class_2=df.loc[df["level"]==2,"image"]

class_3=df.loc[df["level"]==3,"image"]

class_4=df.loc[df["level"]==4,"image"]



#now place the respective images in respective folders
placing_image(class_0,train_0_dir,val_0_dir)
placing_image(class_1,train_1_dir,val_1_dir)
placing_image(class_2,train_2_dir,val_2_dir)
placing_image(class_3,train_3_dir,val_3_dir)
placing_image(class_4,train_4_dir,val_4_dir)
##


#****************************************** END OF TRANSFERING RESPECTIVE IMAGES INTO FOLDERS**********




