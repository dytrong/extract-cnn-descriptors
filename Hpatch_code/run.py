import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image 
from compute_distance_and_match import *
from compute_keypoints_patch import * 
from forward import *
import time
from rgb2hsi import *
from delete_dir import *

#####global variable######
device=torch.device('cuda:1') #调用gpu:1
Model_Img_size=224
Max_kp_num=500
img_suffix='.ppm'
txt_suffix='.h5'
#####将图像格式转换为tensor类型
img_to_tensor = transforms.ToTensor()

#####download models######
start=time.time()
mynet=models.densenet169(pretrained=True).cuda(device)
#不加这行代码，程序预测结果不准
mynet.eval() 
end=time.time()
print('init spend time '+str(end-start))

#############begin class##################
class generate_des:
    def __init__(self,net,img_tensor,mini_batch_size=16,net_type='densenet'):
        self.descriptor=self.extract_batch_conv_features(net,img_tensor,mini_batch_size,net_type)
    #####extract batch conv features#####
    def extract_batch_conv_features(self,net,input_data,mini_batch_size,net_type):
        batch_number=int(len(input_data)/mini_batch_size)
        descriptor_init=self.extract_conv_features(net,input_data[:mini_batch_size],net_type).cpu().detach().numpy()
        for i in range(1,batch_number):
            mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            temp_descriptor=self.extract_conv_features(net,mini_batch,net_type).cpu().detach().numpy()
            descriptor_init=np.vstack((descriptor_init,temp_descriptor))
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
            ####vgg16 is in forward.py
            x=vgg16(net,input_data)
        if net_type.startswith('vgg19'):
            x=vgg19(net,input_data)
        if net_type.startswith('inception_v3'):
            x=inception_v3(net,input_data)
        if net_type.startswith('resnet'):
            x=resnet(net,input_data)
        if net_type.startswith('densenet'):
            x=densenet(net,input_data)
        return x
#######end class, return image descriptors######

#####change images to tensor#####
def change_images_to_tensor(H5_Patch):
    img_list=[]
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
    return img_tensor

#####分批进行计算.如果一起读入，数据太大，读取时间太长，不能充分利用GPU.
def compute_batch_descriptor(net,input_data,mini_batch_size):
    ####计算有多少个mini_batch
    batch_number=int(len(input_data)/mini_batch_size)
    ####计算第一个mini_batch的描述符
    descriptor_init=generate_des(mynet,input_data[:mini_batch_size].cuda(device)).descriptor
    for i in range(1,batch_number):
        mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
        temp_descriptor=generate_des(mynet,mini_batch.cuda(device)).descriptor
        ####将描述符叠起来
        descriptor_init=np.vstack((descriptor_init,temp_descriptor))
    #####avoid the last mini_batch is NULL######
    if (len(input_data)%mini_batch_size==0):
        return descriptor_init
    descriptor=generate_des(net,input_data[mini_batch_size*batch_number:len(input_data)+1].cuda(device)).descriptor
    #####aviod the batch_number=0######
    if batch_number > 0:
        descriptor=np.vstack((descriptor_init,descriptor))
    return descriptor

def compute_patch_descriptor(Img_path,H5_patch_path,mini_batch_size=64):
    ####generate patch img .h5 file, return valid key points
    valid_keypoints=compute_valid_keypoints(Img_path,H5_patch_path,Max_kp_num)
    input_data=change_images_to_tensor(H5_patch_path)
    desc=compute_batch_descriptor(mynet,input_data,mini_batch_size)
    return valid_keypoints,desc    


def compute_mAP(file_path): 
    sum_mAP=0
    extract_desc_time=[]
    compute_desc_dis_time=[]
    for i in range(2,7):
        print("start compute the "+str(i)+" pairs matches")
        base_path='./data/hpatches_sequences_dataset/'+str(file_path)+'/'
        H_path=base_path+'H_1_'+str(i)
        Img_path_A=base_path+str(1)+img_suffix
        Img_path_B=base_path+str(i)+img_suffix
        H5_Patch_A='./data/h5_patch_img/img'+str(1)+txt_suffix
        H5_Patch_B='./data/h5_patch_img/img'+str(i)+txt_suffix
        img1=cv2.imread(Img_path_A)
        img2=cv2.imread(Img_path_B)
        '''
        ##计算光照强度变化，TH为亮度改变大小
        #Th=0
        #####改变img2像素的亮度值，并保持起来
        #img2=change_intensity(img2,Th)
        #Img_path_B='/home/data1/daizhuang/pytorch/data/Intensity_Image/'+str(Th)+'.jpg'
        '''
        start=time.time()
        kp1,desc1=compute_patch_descriptor(Img_path_A,H5_Patch_A)
        kp2,desc2=compute_patch_descriptor(Img_path_B,H5_Patch_B)
        end=time.time()
        diff_time=end-start
        extract_desc_time.append(diff_time)
        #print('提取描述符共耗时:'+str(end-start))
        start=time.time()
        max_match=compute_max_match_cc(desc1,desc2)
        end=time.time()
        diff_time=end-start
        compute_desc_dis_time.append(diff_time)
        #print('计算描述符距离共耗时:'+str(end-start))
        mAP=match(img1,img2,kp1,kp2,max_match,H_path)
        sum_mAP=sum_mAP+mAP
    mAP=sum_mAP/5
    total_time_extract=np.sum(extract_desc_time)/5
    total_time_compute=np.sum(compute_desc_dis_time)/5
    print('提取描述符共耗时:'+str(total_time_extract))
    print('计算描述符距离共耗时:'+str(total_time_compute))
    return mAP

if __name__=="__main__":
    start=time.time()
    all_mAP=[]
    Count=0
    for roots, dirs, files in os.walk('./data/hpatches_sequences_dataset/'):
        for Dir in dirs:
            if Dir[0]=='i':
                print('读取的图像:'+Dir)
                Count=Count+1
                print('读取的图片张数:'+str(Count))
                mAP=compute_mAP(Dir)
                all_mAP.append(mAP)
                print('\n')
    print('所有数据的平均精度为:'+str(np.sum(all_mAP)/len(all_mAP)))
    end=time.time()
    print('总共耗时:'+str(end-start))           
