import cv2
import numpy as np
import h5py
import math
import time
from matplotlib import pyplot as plt
import torch
from sklearn import preprocessing
#from autoencoder import autoencoder
#from sklearn.decomposition import PCA
device=torch.device('cuda:1')
#################################
###### PCA降维######
##### des is a numpy.array###
def PCA(des):
    pca=PCA(n_components=128)
    des=pca.fit_transform(des1)
    return des
###计算L2_dis_matrix#######
###des1:des1 is a numpy.array####
###des2:des2 is a numpy.array####
def compute_des_L2_dis(des1,des2):
    #des1=PCA(des1)
    #des2=PCA(des2)
    #des1=autoencoder(des1)
    #des2=autoencoder(des2)
    #des1=np.sqrt(preprocessing.normalize(des1,norm='l2'))
    #des2=np.sqrt(preprocessing.normalize(des2,norm='l2')) 
    des1=torch.from_numpy(des1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    des2=torch.from_numpy(des2).cuda(device)
    pdist=torch.nn.PairwiseDistance(2)
    dis_matrix=np.zeros((len(des1),len(des2)))
    dim=des1.shape[1]
    for i in range(len(des1)):
        dis_matrix[i]=pdist(des1[i].view(1,dim),des2)
    return dis_matrix

###计算cos_dis_matrix#######
###des1:des1 is a numpy.array####
###des2:des2 is a numpy.array####
def compute_des_cos_dis(des1,des2):
    des1=torch.from_numpy(des1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    des2=torch.from_numpy(des2).cuda(device)
    des1_T=torch.transpose(des1,0,1) #### 转置
    des2_T=torch.transpose(des2,0,1) 
    temp_1=torch.mm(des1,des2_T).cpu().numpy() #### torch.mm 矩阵的点乘 ，data.cpu().numpy() 将gpu的tensor 转化为cpu的numpy
    temp_2=torch.pow(torch.mm(des1,des1_T),0.5).cpu().numpy()
    temp_3=torch.pow(torch.mm(des2,des2_T),0.5).cpu().numpy()
    temp_matrix=np.zeros((temp_2.shape[0],temp_3.shape[0])) #####初始化矩阵
    for i in range(temp_2.shape[0]):
        for j in range(temp_3.shape[0]):
            temp_matrix[i,j]=temp_2[i,i]*temp_3[j,j] ####取对角线元素相乘
    cos_dis_matrix=temp_1/temp_matrix     #####cosine distance
    return cos_dis_matrix


def compute_max_match_nn(des1,des2):
    cos_dis=compute_des_L2_dis(des1,des2)    ####L2_dis
    iterate_left=cos_dis.shape[0]
    Max_Matches=np.dtype({'names':['i','j','MAX'],'formats':['i','i','f']})
    M1=np.zeros(iterate_left,dtype=Max_Matches) ###初始化保存从左到右最佳匹配的矩阵
    MAX_XY=[]
    #compute the max match from left to right
    for i in range(iterate_left):
        MAX_COS_DIS=np.min(cos_dis[i,:])   ####L2_dis
        M1[i]['i']=i
        M1[i]['j']=np.argmin(cos_dis[i,:])  ####L2_dis
        M1[i]['MAX']=MAX_COS_DIS
        MAX_XY.append([M1[i]['i'],M1[i]['j'],M1[i]['MAX']])
    temp_list=(M1,cos_dis)
    return temp_list


def compute_max_match_nn_cos(des1,des2):
    cos_dis=compute_des_cos_dis(des1,des2)  ####cos_dis
    iterate_left=cos_dis.shape[0]
    Max_Matches=np.dtype({'names':['i','j','MAX'],'formats':['i','i','f']})
    M1=np.zeros(iterate_left,dtype=Max_Matches) ###初始化保存从左到右最佳匹配的矩阵
    MAX_XY=[]
    #compute the max match from left to right
    for i in range(iterate_left):
        MAX_COS_DIS=np.max(cos_dis[i,:])  ####cos_dis
        M1[i]['i']=i
        M1[i]['j']=np.argmax(cos_dis[i,:]) ####cos_dis
        M1[i]['MAX']=MAX_COS_DIS
    temp_list=(M1,cos_dis)
    return temp_list

def compute_max_match_cc(des1,des2,method='L2'):
    start=time.time()
    (M1,cos_dis)=compute_max_match_nn(des1,des2)
    iterate_left=cos_dis.shape[0]
    iterate_right=cos_dis.shape[1]
    M2=np.zeros(iterate_right,dtype=M1.dtype)
    #compute the max match from right to left
    if method=='L2':
        for j in range(iterate_right):
            MAX_COS_DIS=np.min(cos_dis[:,j])
            M2[j]['MAX']=MAX_COS_DIS
            M2[j]['i']=j
            M2[j]['j']=np.argmin(cos_dis[:,j])
    else:
        for j in range(iterate_right):
            MAX_COS_DIS=np.max(cos_dis[:,j])
            M2[j]['MAX']=MAX_COS_DIS
            M2[j]['i']=j
            M2[j]['j']=np.argmax(cos_dis[:,j])
    MAX_XY=[]
    for i in range(iterate_left):
       if M2[M1[i]['j']]['j']==i:
         MAX_XY.append([M1[i]['i'],M1[i]['j'],M1[i]['MAX']])
    MAX_XY=np.array(MAX_XY)
    end=time.time()
    print("compute max match use cc spend total time "+str(end-start)+" seconds")
    return MAX_XY

def append_image(img1,img2):
  rows1=img1.shape[0]
  rows2=img2.shape[0]
  if rows1<rows2:
      concat=np.zeros((rows2-rows1,img1.shape[1],3),dtype=np.uint8)
      img1=np.concatenate((img1,concat),axis=0)
  if rows1>rows2:
      concat=np.zeros((rows1-rows2,img2.shape[1],3),dtype=np.uint8)
      img2=np.concatenate((img2,concat),axis=0)
  img3=np.concatenate((img1,img2), axis=1)
  return img3

def bgr_rgb(img):
    (b,g,r)=cv2.split(img)
    return cv2.merge([r,g,b])
    
def match(img1,img2,kp1,kp2,max_match,count,save_path):
    cols1 = img1.shape[1]
    img3=append_image(img1,img2)
    if len(max_match)==0:
        cv2.imwrite(save_path+str(count)+'.ppm',img3)
        return 0,0. 
    key_points1=[]
    key_points2=[]
    for i in range(len(max_match)):
        key_points1.append(kp1[int(max_match[i][0])])
        key_points2.append(kp2[int(max_match[i][1])])
    ####ransac find H matrix####
    key_points1=np.float32(key_points1)
    key_points2=np.float32(key_points2)
    #print('matches number before ransac '+str(len(key_points1)))
    total_number=len(key_points1)
    print('经过ransac之前的匹配对:'+str(total_number))
    inliner_number=0
    if(len(key_points1)>=4):
        H,mask=cv2.findHomography(key_points1,key_points2,cv2.RANSAC)
        #F, mask = cv2.findFundamentalMat(key_points1, key_points2, cv2.RANSAC)
        key_points1=key_points1[mask.ravel()==1]
        key_points2=key_points2[mask.ravel()==1]
        inliner_number=len(key_points1)
    print('inliner 数量为:'+str(inliner_number))
    inliner_radio=float(inliner_number)/total_number
    print('内点率为:'+str(inliner_radio))
    #return inliner_number,inliner_radio
    for i in range(len(key_points1)):
        (x1,y1)=key_points1[i]
        (x2,y2)=key_points2[i]
        cv2.circle(img3,(int(np.round(x1)),int(np.round(y1))),2,(0,255,255),2)
        cv2.circle(img3,(int(np.round(x2)+cols1),int(np.round(y2))),2,(0,0,255),2)
        cv2.line(img3, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (255, 0, 0), 2, lineType=cv2.LINE_AA, shift=0)
    cv2.imwrite(save_path+str(count)+'.jpg',img3)
    cv2.imshow("img3",img3)
    cv2.waitKey(0)
    return inliner_number,inliner_radio
