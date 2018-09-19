import cv2
import numpy as np
from compute_distance_and_match import *
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

def compute_mAP(illu_path):
    sum_mAP=0
    for i in range(4,5):
        start=time.time()
        Max_kp_num=500
        img_suffix='.ppm'
        print("start compute the "+str(i)+" pairs matches")
        base_path='/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/'+str(illu_path)+'/'
        H_path=base_path+'H_1_'+str(i)
        Img_path_A=base_path+str(1)+img_suffix
        Img_path_B=base_path+str(i)+img_suffix
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
        print('extract conv features spend time '+str(end-start))
        start=time.time()
        max_match=compute_max_match_cc(desc1,desc2)
        mAP=match(img1,img2,kp1_list,kp2_list,max_match,H_path)
        sum_mAP=sum_mAP+mAP
        end=time.time()
        print('compute distance and match spend total time '+str(end-start))
    mAP=sum_mAP/5
    return mAP

if __name__=="__main__":
    start=time.time()
    all_mAP=[]
    Count=0
    for roots, dirs, files in os.walk('/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/'):
        for dir in dirs:
            if dir=='i_kions':
                print('读取的图像:'+dir)
                Count=Count+1
                print('读取的图片张数:'+str(Count))
                mAP=compute_mAP(dir)
                print('\n')
                all_mAP.append(mAP)
    total=np.sum(all_mAP)
    average=float(total)/len(all_mAP)            
    print('所有数据的平均精度为:'+str(average))
    end=time.time()
    print('总共耗时:'+str(end-start)) 
