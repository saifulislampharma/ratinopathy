#import numpy as np
import pandas as pd
import os
import shutil as sh
from . import config

#directories to move extra 0 category image
old_dir_train=os.path.join(config.data_path(),"eyepack","train","0")
new_dir_train=os.path.join(config.data_path(),"data_backup","train","0")
old_dir_val=os.path.join(config.data_path(),"eyepack","validation","0")
new_dir_val=os.path.join(config.data_path(),"data_backup","validation","0")
#os.makedirs(new_dir,exist_ok=True)

##print(old_dir)
##print(new_dir)


##count=config.total_count_files(old_dir)
##print("before moving total count of 0 category was :"+str(count))
##
###after moving
##files_list=os.listdir(old_dir)
##movable_files_count=int(count*.5)
##print(str(movable_files_count))
##movable_files=files_list[:movable_files_count]
##for image in movable_files:
##    input_dirs=os.path.join(old_dir,image)
##    sh.move(input_dirs,new_dir)
##
##count_move=config.total_count_files(old_dir)
##print("after moving total count of 0 category was :"+str(count_move))
##
##count_new=config.total_count_files(new_dir)
##print("in new destination 0 category was :"+str(count_new))


#function for transferring from one directory to another directory

def move_files(old_dirs,new_dirs):
    files_list=os.listdir(old_dirs)
    for image in files_list:
        input_dirs=os.path.join(old_dirs,image)
        sh.move(input_dirs,new_dirs)


###transfering all files from data_backup/train/0 to eyepack/train/0
##move_files(new_dir_train,old_dir_train)
###transfering all files from data_backup/validation/0 to eyepack/validation/0
##move_files(new_dir_val,old_dir_val)


print("after transfering file...")
print("in train/0 directory  "+str(config.total_count_files(old_dir_train))+" image found")
print("in validation/0 directory  "+str(config.total_count_files(old_dir_val))+" image found")






