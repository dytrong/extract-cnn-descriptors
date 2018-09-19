import cv2
import numpy as np
from compute_distance_and_match_ransac import *
import time
import os

def sift_detect(img1,img2,Max_kp_num,detector='surf'):
    start=time.time()
    if detector.startswith('sift'):
        print("sift detector ......")
        detector=cv2.xfeatures2d.SIFT_create(Max_kp_num)
    else:
        print('surf detector ......')
        detector=cv2.xfeatures2d.SURF_create(Max_kp_num)
    kp1,des1=detector.detectAndCompute(img1,None)
    kp2,des2=detector.detectAndCompute(img2,None)
    return kp1,des1,kp2,des2
if __name__=="__main__":
    Sum_inliner_radio=0.
    Sum_inliner=0.
    img_suffix='.jpg'
    Max_kp_num=500
    txt_suffix='.h5'
    for i in range(255,256):
        start=time.time()
        print("start compute the "+str(i)+" pairs matches")
        Img_path_A='../data/UACampus_data/2215/'+str(i)+img_suffix
        Img_path_B='../data/UACampus_data/0620/'+str(i)+img_suffix
        H5_Patch_A='../data/UACampus_data/h5_patch_A/img'+str(i)+txt_suffix
        H5_Patch_B='../data/UACampus_data/h5_patch_B/img'+str(i)+txt_suffix
        save_path='../data/results/'
        img1=cv2.imread(Img_path_A)
        img2=cv2.imread(Img_path_B)
        kp1,desc1,kp2,desc2=sift_detect(img1,img2,Max_kp_num,detector='sift')
        kp1_list=[]
        kp2_list=[]
        for i in range(len(kp1)):
            kp1_list.append(kp1[i].pt)
        for j in range(len(kp2)):
            kp2_list.append(kp2[j].pt)
        end=time.time()
        print('extract handcraft features spend time '+str(end-start))
        start=time.time()
        max_match=compute_max_match_cc(desc1,desc2)
        inliner,inliner_radio=match(img1,img2,kp1_list,kp2_list,max_match,i,save_path)
        Sum_inliner_radio=inliner_radio+Sum_inliner_radio
        Sum_inliner=inliner+Sum_inliner
        end=time.time()
        print('compute distance and match spend total time '+str(end-start))
        print('\n')
    print('平均内点率为：'+str(Sum_inliner_radio/285))
    print('平均内点数量:'+str(Sum_inliner/285))
