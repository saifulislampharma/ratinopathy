import os

image_size = (299,299,3)
image_size_gen = (299,299)
num_classes = 5
batch_size = 16
epochs = 30


def root_path():
    return os.path.dirname(__file__)

def data_path():
    return os.path.join(root_path(),"data")

def dataset_path():
    return os.path.join(root_path(),"dataset")

def preprocessing_path():
    return os.path.join(root_path(),"preprocessing")

def src_path():
    return os.path.join(root_path(),"src")

def output_path():
    return os.path.join(root_path(),"output")

def weight_path():
    return os.path.join(root_path(),"weight")




#data dir

INPUT_DIR = os.path.join(dataset_path(),"eyepack")

BASE_DIR=os.path.join(data_path(),"eyepack")

#train directory creation
TRAIN_DIR=os.path.join(BASE_DIR,"train")
os.makedirs(TRAIN_DIR,exist_ok=True)

#validation directory creation
VAL_DIR=os.path.join(BASE_DIR,"validation")
os.makedirs(VAL_DIR,exist_ok=True)

#test directory creation
TEST_DIR=os.path.join(BASE_DIR,"test")
os.makedirs(TEST_DIR,exist_ok=True)



def total_count_files(dir):
    """ count all files in a directory recursively
    """
    total = 0
    for root, dirs, files in os.walk(dir):
        total += len(files)
    return total

