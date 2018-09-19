from compute_distance_and_match_v3 import append_image
import cv2
import time
import os
def compute_mAP(illu_path,count):
    sum_mAP=0
    #base_path='/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/'+str(illu_path)+'/'
    Img_path_A=illu_path+str(2)+'.jpg' 
    img1=cv2.imread(Img_path_A)
    for i in [100,150,200,300,350]:
        start=time.time()
        print("start compute the "+str(i)+" pairs matches")
        Img_path_B=illu_path+str(i)+'.jpg'
        img2=cv2.imread(Img_path_B)
        img1=append_image(img1,img2)
    cv2.imwrite('/home/data1/daizhuang/pytorch/data/results/'+str(count)+'.jpg',img1)


if __name__=="__main__":
    '''
    start=time.time()
    all_mAP=[]
    count=0
    for roots, dirs, files in os.walk('/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/'):
        for dir in dirs:
            if dir[0]=='i':
                compute_mAP(dir,count)
                count=count+1
    '''
    compute_mAP('/home/data1/daizhuang/pytorch/data/Uacampus/BY/',0)
