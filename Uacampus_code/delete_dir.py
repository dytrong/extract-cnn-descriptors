import os
def make_dir(Path,dir_thresh):
  for i in range(dir_thresh):
    file_path=Path+"image_patch_"+str(i)
    if not os.path.exists(file_path):
      os.mkdir(file_path)

def delete_dir(path):
    ls=os.listdir(path) #return the subfile under the path 
    for i in ls:
        c_path=os.path.join(path,i) #combination the father_path and sub_path
        if os.path.isdir(c_path):
            delete_dir(c_path)
        else:
            os.remove(c_path)

path_1='./data/h5_patch_copy'
path_2='./data/results'
path_list=[path_1,path_2]

for i in path_list:
    print('delete '+str(i))
    delete_dir(i)
