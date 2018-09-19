import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
from compute_distance_and_match import *
from compute_keypoints_patch import * 
from forward import *
import time
from delete_dir import *
import sys
sys.path.append('/home/data/daizhuang/tensorflow_model/research/slim')
from datasets import imagenet
#####global variable######
device=torch.device('cuda:1') #调用gpu:1
Model_Img_size=224
#####download models######
start=time.time()
myresnet=models.vgg16(pretrained=True)
#print(myresnet)
myresnet.eval() #不加这行代码，程序预测结果错误:
end=time.time()
print('init spend time '+str(end-start))
img_to_tensor = transforms.ToTensor()
class generate_des:
    def __init__(self,net,image_path,Model_Img_size):
        self.img_tensor=self.change_images_to_tensor(image_path,Model_Img_size)
        self.descriptor=self.extract_conv_features(net,self.img_tensor)
    #####extract conv features#####
    def extract_conv_features(self,net,input_data,net_type='vgg16'):
        if net_type.startswith('vgg16'):
            x=vgg16(net,input_data)  ####vgg16 is in forward.py
        if net_type.startswith('resnet'):
            x=resnet(net,input_data)
        if net_type.startswith('densenet'):
            x=densenet(net,input_data)
        if net_type.startswith('inception_v3'):
            x=inception_v3(net,input_data)
        return x    
    #####change images to tensor#####
    def change_images_to_tensor(self,image_path,Model_Img_size):
        img_list=[]
        for i in range(5):
            _image_path=image_path+str(i)+'.ppm'
            img=cv2.imread(_image_path)
            img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))###change image format from cv2 to Image  
            img=img.resize((Model_Img_size,Model_Img_size))
            #img=np.array(img,dtype=np.float32)
            img=img_to_tensor(img)
            img=img.numpy()
            img=img.reshape(3,Model_Img_size,Model_Img_size)
            img_list.append(img)
        img_array=np.array(img_list)
        img_tensor=torch.from_numpy(img_array)
        '''
        _image_path=image_path+'171.jpg'
        img=Image.open(_image_path)
        img=img.resize((Model_Img_size,Model_Img_size))
        tensor=img_to_tensor(img)
        tensor=tensor.resize_(1,3,Model_Img_size,Model_Img_size)
        #print(tensor)
        '''
        return img_tensor


if __name__=="__main__":
    Img_path_A='./data/AY_pre/'
    desc=generate_des(myresnet,Img_path_A,Model_Img_size).descriptor.detach().numpy()
    max_index=np.argmax(desc[0,:])
    print(max_index)
    print(desc[0,:])
    names = imagenet.create_readable_names_for_imagenet_labels()
    print(names[max_index+1])


