import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
from compute_distance_and_match_uac import *
from compute_keypoints_patch import * 
from forward import *
from rgb2hsi import *
import time
#####global variable######
device=torch.device('cuda:0') #调用gpu:1
Model_Img_size=224
mini_batch_size=32
Max_kp_num=500
img_suffix='.ppm'
txt_suffix='.h5'
img_to_tensor = transforms.ToTensor()
#####download models######
start=time.time()
myresnet=models.alexnet(pretrained=True).cuda(device)
#print(myresnet)
#不加这行代码，程序预测结果错误:
myresnet.eval()
end=time.time()
print('init spend time '+str(end-start))

#####class#########
class generate_des:
    def __init__(self,net,img_tensor,mini_batch_size=8,net_type='alexnet'):
        self.descriptor=self.extract_batch_conv_features(net,img_tensor,mini_batch_size,net_type)
    #####extract batch conv features#####
    def extract_batch_conv_features(self,net,input_data,mini_batch_size,net_type):
        batch_number=int(len(input_data)/mini_batch_size)
        descriptor_init=self.extract_conv_features(net,input_data[:mini_batch_size],net_type).cpu().detach().numpy()
        #start=time.time()
        for i in range(1,batch_number):
            mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            temp_descriptor=self.extract_conv_features(net,mini_batch,net_type).cpu().detach().numpy()
            descriptor_init=np.vstack((descriptor_init,temp_descriptor))
        #end=time.time()
        #print('加载数据耗时:'+str(end-start))
        #####avoid the last mini_batch is NULL######
        if (len(input_data)%mini_batch_size==0):
            return descriptor_init
        descriptor=self.extract_conv_features(net,input_data[mini_batch_size*batch_number:len(input_data)+1],net_type).cpu().detach().numpy()
        #####aviod the batch_number=0######
        if batch_number > 0:
            descriptor=np.vstack((descriptor_init,descriptor))
        return descriptor
    #####extract conv features#####
    def extract_conv_features(self,net,input_data,net_type):
        if net_type.startswith('alexnet'):
            x=alexnet(net,input_data)
        if net_type.startswith('vgg16'):
            x=vgg16(net,input_data)  ####vgg16 is in forward.py
        if net_type.startswith('vgg19'):
            x=vgg16(net,input_data)  ####vgg16 is in forward.py
        if net_type.startswith('inception_v3'):
            x=inception_v3(net,input_data)
        if net_type.startswith('resnet'):
            x=resnet(net,input_data)
        if net_type.startswith('densenet'):
            x=densenet(net,input_data)
        return x

#####change images to tensor#####
def change_images_to_tensor(H5_Patch,Model_Img_size=224):
    img_list=[]
    start=time.time()
    #the patch image .h5 file
    Img_h5=h5py.File(H5_Patch,'r') 
    for i in range(len(Img_h5)):
        img=Img_h5[str(i)][:]
        ###change image format from cv2 to Image
        img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        img=img.resize((Model_Img_size,Model_Img_size))
        img=img_to_tensor(img)
        img=img.numpy()
        img=img.reshape(3,Model_Img_size,Model_Img_size)
        img_list.append(img)
    img_array=np.array(img_list)
    img_tensor=torch.from_numpy(img_array)
    end=time.time()
    print('读取图片耗时:'+str(end-start))
    return img_tensor

#####分批进行计算.如果一起读入，数据太大，读取时间太长，不能充分利用GPU.
def compute_batch_descriptor(net,input_data,mini_batch_size):
    batch_number=int(len(input_data)/mini_batch_size)
    descriptor_init=generate_des(myresnet,input_data[:mini_batch_size].cuda(device)).descriptor ####第一个mini_batch
    #start=time.time()
    for i in range(1,batch_number):
        mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
        temp_descriptor=generate_des(myresnet,mini_batch.cuda(device)).descriptor
        descriptor_init=np.vstack((descriptor_init,temp_descriptor)) ####将描述符叠起来
    #end=time.time()
    #print("计算描述符共耗时:"+str(end-start))
    #####avoid the last mini_batch is NULL######
    if (len(input_data)%mini_batch_size==0):
        return descriptor_init
    descriptor=generate_des(net,input_data[mini_batch_size*batch_number:len(input_data)+1].cuda(device)).descriptor
    #####aviod the batch_number=0######
    if batch_number > 0:
        descriptor=np.vstack((descriptor_init,descriptor))
    return descriptor

def compute_patch_descriptor(Img_path,H5_patch_path,mini_batch_size=32):
    valid_keypoints=compute_valid_keypoints(Img_path,H5_patch_path,Max_kp_num) ####generate patch img .h5 file#####
    input_data=change_images_to_tensor(H5_patch_path)
    desc=compute_batch_descriptor(myresnet,input_data,mini_batch_size)
    return valid_keypoints,desc    

if __name__=="__main__":
    Sum_inliner_radio=0
    Sum_inliner=0
    for i in range(2,3):
        start=time.time()
        print("start compute the "+str(i)+" pairs matches")
        Img_path_A='./data/Intensity/AY/'+str(i)+img_suffix
        Img_path_B='./data/Intensity/BY/'+str(i)+img_suffix
        H5_Patch_A='./data/Intensity/h5_patch_A/img'+str(i)+txt_suffix
        H5_Patch_B='./data/Intensity/h5_patch_B/img'+str(i)+txt_suffix
        save_path='./data/results/'
        img1=cv2.imread(Img_path_A)
        img2=cv2.imread(Img_path_B)
        '''
        Th=-10
        img2=change_intensity(img2,Th)
        Img_path_B='/home/data1/daizhuang/pytorch/data/Intensity_Image/'+str(Th)+'.jpg'
        '''
        kp1,desc1=compute_patch_descriptor(Img_path_A,H5_Patch_A)
        kp2,desc2=compute_patch_descriptor(Img_path_B,H5_Patch_B)
        end=time.time()
        print('extract conv features spend time '+str(end-start))
        start=time.time()
        max_match=compute_max_match_cc(desc1,desc2)
        inliner,inliner_radio=match(img1,img2,kp1,kp2,max_match,i,save_path)
        Sum_inliner_radio=inliner_radio+Sum_inliner_radio
        Sum_inliner=inliner+Sum_inliner
        end=time.time()
        print('compute distance and match spend total time '+str(end-start))
        print('\n')
    print('平均内点率为：'+str(Sum_inliner_radio/5))
    print('平均内点数量:'+str(Sum_inliner/5))
