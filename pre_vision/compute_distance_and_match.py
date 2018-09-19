import cv2
import numpy as np
import h5py
import math
import time
from matplotlib import pyplot as plt
import torch
from sklearn import preprocessing
#from autoencoder import autoencoder
from sklearn.decomposition import PCA
device=torch.device('cuda:1')
#################################
#####图像拼接#####
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
#####图像由bgr格式转为rgb格式
def bgr_rgb(img):
    (b,g,r)=cv2.split(img)
    return cv2.merge([r,g,b])
###### PCA降维和白化######
##### des is a numpy.array###
def pca_whiten(input_data):
    pca = PCA(n_components=input_data.shape[1],whiten=True)
    pca = pca.fit(input_data)
    out_pca= pca.transform(input_data)
    return out_pca
###计算L2_dis_matrix#######
###des1:des1 is a numpy.array####
###des2:des2 is a numpy.array####
def compute_des_L2_dis(des1,des2):
    #des1=pca_whiten(des1)
    #des2=pca_whiten(des2)
    #des1=autoencoder(des1)
    #des2=autoencoder(des2)
    #des1=preprocessing.normalize(des1,norm='l2')
    #des2=preprocessing.normalize(des2,norm='l2') 
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

#######计算像素的欧式距离,专用的一个函数
def compute_pixs_distance(des1,des2):
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
    MAX_XY=np.array(MAX_XY)
    return MAX_XY

def compute_max_match_nn_L2(des1,des2):
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

def compute_max_match_cc(des1,des2,method='cos'):
    start=time.time()
    if method=='L2':
        (M1,cos_dis)=compute_max_match_nn_L2(des1,des2)
    if method=='cos':
        (M1,cos_dis)=compute_max_match_nn_cos(des1,des2)
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
    if method=='cos':
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

#########通过H矩阵算左图到右图对应位置
def compute_corresponse(pt1,H_path):
    H=np.loadtxt(H_path,dtype=np.float32)
    pt2_list=[]
    for i in range(len(pt1)):
        (x1,y1)=pt1[i]
        x2=(H[0][0]*x1+H[0][1]*y1+H[0][2])/(H[2][0]*x1+H[2][1]*y1+H[2][2])
        y2=(H[1][0]*x1+H[1][1]*y1+H[1][2])/(H[2][0]*x1+H[2][1]*y1+H[2][2])
        pt2=(x2,y2)
        pt2_list.append(pt2)
    pt2_array=np.array(pt2_list)
    return pt2_array
#########计算两幅图像中实际匹配点对
def points_distance(pt1,pt2,H_path):
    #####计算图1进过H变换后在图2中的估计位置
    pt1=compute_corresponse(pt1,H_path)
    pt1=np.array(pt1)
    pt2=np.array(pt2)
    #####计算图1估计出的位置与图二实际位置的欧式距离
    L2_dis=compute_pixs_distance(pt1,pt2) 
    valid_match=[]
    #####找出实际真正的匹配对
    for i in range(len(L2_dis)):
        ####当欧式距离小于1.5像素，认为是同一个点
        if L2_dis[i][2]<=3:
            valid_match.append(L2_dis[i])  
    valid_match=np.array(valid_match)     
    return valid_match 

def compute_valid_match(max_match,Thresh,method):
    valid_match=[]
    for i in range(len(max_match)):
        ######当距离公式选用L2范数时，阈值是小于等于
        if max_match[i][2]<=Thresh and method=='L2':
            valid_match.append(max_match[i])
        #####当距离公式选用cos距离时,阈值是大于等于
        if max_match[i][2]>=Thresh and method=='cos':
            valid_match.append(max_match[i])
    valid_match=np.array(valid_match)
    return valid_match

def compute_PR(max_match,thresh,ground_true_match,imshow,method):
    max_match=compute_valid_match(max_match,thresh,method)
    correct_match_number=0
    correct_match=[]
    detect_match_number=len(max_match)
    ground_true_match_number=len(ground_true_match)
    if ground_true_match_number==0:
        return 0,0 
    for i in range(detect_match_number):
        ####看检测出的匹配点是否在ground_true_match中,若在则返回它的位置
        if max_match[i][0] in list(ground_true_match[:,0]):
            #####numpy.array没有index这个属性
            index=list(ground_true_match[:,0]).index(max_match[i][0])
            if max_match[i][1]==ground_true_match[index][1]:
                 correct_match_number=correct_match_number+1
                 correct_match.append(max_match[i])
    if detect_match_number==0:
        return 0,0
    recall=float(correct_match_number)/ground_true_match_number
    precision=float(correct_match_number)/detect_match_number
    if imshow:
        return precision,recall,correct_match,max_match
    else:
        return precision,recall

def match(img1,img2,kp1,kp2,max_match,H_path,imshow=False,method='cos'):
    if imshow:
        #####显示匹配对
        ground_true_match=points_distance(kp1,kp2,H_path)
        if len(max_match)==0:
            print('不存在匹配对')
            return 0
        max_list=max_match[:,2]
        max_list.sort()
        if method=='L2':
            Thresh=max_list[19]
        if method=='cos':
            Thresh=max_list[-20]
        #######max_match为算法检测出的匹配对,
        #######correct_match为检测出的算法为真实的匹配对
        p,r,correct_match,max_match=compute_PR(max_match,Thresh,ground_true_match,imshow,method)
        print('正确率为:'+str(p))
        print(len(ground_true_match))
        show_keypoints(kp1,kp2,img1,img2,correct_match,max_match) 
    else:
        ground_true_match=points_distance(kp1,kp2,H_path) ####实际匹配对
        max_number=np.max(max_match[:,2])
        min_number=np.min(max_match[:,2])
        step_number=50
        P_list=[]
        R_list=[]
        AP=[]
        for i in range(step_number):
            thresh=max_number-((max_number-min_number)/step_number)*i
            precision,recall=compute_PR(max_match,thresh,ground_true_match,imshow,method)
            P_list.append(precision)
            R_list.append(recall)
        #####当选用L2范数时，计算出的精度是小到大的，
        #####这样我们算PR面积的时候公式就和cos算出的不统一了
        #####所以将PR list反转一下，和cos计算出来的就一致了
        if method=='L2':
            P_list.reverse() ###将列表反转
            R_list.reverse() 
        for i in range(1,step_number):
            AP.append(P_list[i]*(R_list[i]-R_list[i-1]))
        mAP=np.sum(AP)
        print("数据的mAP为:"+str(mAP))
        return mAP

def show_keypoints(kp1,kp2,img1,img2,detect_correct_match,max_match):
    cols1 = img1.shape[1]
    img3=append_image(img1,img2)
    if len(max_match)==0:
        return
    key_points1=[]
    key_points2=[]
    kp_1=[]
    kp_2=[]
    #####画所有算法认为是匹配上的点
    for i in range(len(max_match)):
        kp_1.append(kp1[int(max_match[i][0])])
        kp_2.append(kp2[int(max_match[i][1])])
    #####实际为真实匹配的点
    for i in range(len(detect_correct_match)):
        key_points1.append(kp1[int(detect_correct_match[i][0])])
        key_points2.append(kp2[int(detect_correct_match[i][1])]) 
    for i in range(len(kp_1)):
        (x1,y1)=kp_1[i]
        (x2,y2)=kp_2[i]
        cv2.circle(img3,(int(np.round(x1)),int(np.round(y1))),1,(0,255,255),2)
        cv2.circle(img3,(int(np.round(x2)+cols1),int(np.round(y2))),1,(0,155,255),2)
        if kp_1[i] in key_points1:
            cv2.line(img3, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (255, 0, 0), 2, lineType=cv2.LINE_AA, shift=0)
        else:
            cv2.line(img3, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (0, 0, 255), 2, lineType=cv2.LINE_AA, shift=0)
    cv2.imwrite('/home/data1/daizhuang/pytorch/data/results/cnn_match.jpg',img3)
    cv2.imshow("img3",img3)
    cv2.waitKey(0)
