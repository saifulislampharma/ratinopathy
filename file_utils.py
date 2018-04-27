#import numpy as np
import pandas as pd
import os
import shutil as sh
from . import config


old_dir=os.path.join(config.data_path(),"eyepack","train","0")
new_dir=os.path.join(config.data_path(),"data_backup","train","0")
os.makedirs(new_dir,exist_ok=True)
##print(old_dir)
##print(new_dir)


count=config.total_count_files(old_dir)
print("before moving total count of 0 category was :"+str(count))

#after moving
files_list=os.listdir(old_dir)
movable_files_count=int(count*.5)
print(str(movable_files_count))
movable_files=files_list[:movable_files_count]
for image in movable_files:
    input_dirs=os.path.join(old_dir,image)
    sh.move(input_dirs,new_dir)

count_move=config.total_count_files(old_dir)
print("after moving total count of 0 category was :"+str(count_move))

count_new=config.total_count_files(new_dir)
print("in new destination 0 category was :"+str(count_new))
