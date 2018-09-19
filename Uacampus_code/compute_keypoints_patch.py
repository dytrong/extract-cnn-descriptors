import cv2
import numpy as np
import os
import h5py

######提取图像的sift特征点
def sift_detect(img,Max_kp_num):
    #change the image format to gray
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #extract sift features
    sift=cv2.xfeatures2d.SIFT_create(Max_kp_num)
    #detect the image keypoints
    keypoints=sift.detect(gray,None)
    return keypoints
######计算有效的特征点块，因为有些特征点在图像边缘地方，以该点为中心，取特征点块可能超过了图像的边界，要舍弃改特征点.
####Image_path:要匹配图像的路径
####patch_img_h5_path:将特征点块存储的.h5文件的路径
####Max_kp_num:检测的sift特征点最多个数
def compute_valid_keypoints(Image_path,patch_img_h5_path,Max_kp_num):
    img=cv2.imread(Image_path)
    keypoints=sift_detect(img,Max_kp_num)
    #generate the pathes depend on keypoints  
    valid_keypoints=generate_keypoints_patch(keypoints,img,patch_img_h5_path)
    print("the detect valid keypoints number "+str(len(valid_keypoints)))
    return valid_keypoints

#####计算特征点所在的图像金字塔层数
def unpack_octave(octave):
    octave=octave&255
    octave=octave if octave<128 else (-128|octave)
    return octave


###keypoints:extract from image by sift
###img:the original image
###patch_image_h5_path:patch image save h5 path
def generate_keypoints_patch(keypoints,img,patch_image_h5_path):
    image_index=0
    F_h5=h5py.File(patch_image_h5_path,'w')
    #patch_size_list=(16,32,64,128,256,512,1024,2048,4096)
    patch_size_list=(32,32,32,32,32,32,32,32)
    #patch_size_list=(64,64,64,64,64,64,64,64)
    #patch_size_list=(128,128,128,128,128,128,128,128)
    diff_keypoints_list=[]
    for k in keypoints:
        #because the image axis is defferent from matrix axis
        x=int(k.pt[1])
        y=int(k.pt[0])
        #unpack the octave
        size_index=unpack_octave(k.octave)+1
        patch_size=patch_size_list[size_index]
        #judgment the boundary
        if (x-patch_size)>0 and (y-patch_size)>0 and (x+patch_size)<img.shape[0] and (y+patch_size)<img.shape[1] and (k.pt not in diff_keypoints_list):
            #delete the same keypoints
            diff_keypoints_list.append(k.pt)
            #the image of keypoint field
            img_patch=img[x-patch_size:x+patch_size,y-patch_size:y+patch_size]
            F_h5[str(image_index)]=img_patch
            image_index=image_index+1
    F_h5.close()
    #return valid different keypoints
    return diff_keypoints_list

