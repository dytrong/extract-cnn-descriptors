import matplotlib.pyplot as plt
'''
###AlexNet
x=['conv2','pool2','conv3','pool3','conv4','pool4','conv5','pool5']
y1=[0.2807,0.3504,0.3485,0.3808,0.3601,0.3725,0.3734,0.3308]
y2=[0.6745,0.6686,0.6972,0.6589,0.7057,0.6805,0.6561,0.6207]
y3=[0.4776,0.5059,0.5227,0.5411,0.5329,0.5558,0.5148,0.5165]

'''
'''
##vgg16
x=['conv3','pool3','conv4','pool4','conv5','pool5']
y1=[0.2589,0.3449,0.3067,0.3935,0.3227,0.3435]
y2=[0.7570,0.7910,0.7457,0.7422,0.5929,0.5526]
y3=[0.5048,0.5680,0.5262,0.5679,0.4578,0.4481]
'''

'''
####resnet101
x=['block1','pool1','block2','pool2','block3','pool3','block4','pool4']
y1=[0.1739,0.2435,0.1675,0.3687,0.3663,0.4245,0.3015,0.2530]
y2=[0.6123,0.6350,0.7209,0.7695,0.7199,0.6952,0.5511,0.4354]
y3=[0.3931,0.4393,0.4442,0.5691,0.5431,0.5599,0.4263,0.3442]

'''
####densenet169
x=['block1','pool1','block2','pool2','block3','pool3','block4','pool4']
y1=[0.0800,0.2576,0.2593,0.3832,0.3378,0.4316,0.3191,0.3150]
y2=[0.5728,0.6981,0.7608,0.7523,0.7258,0.7230,0.6340,0.5764]
y3=[0.3264,0.4779,0.5101,0.5678,0.5318,0.5773,0.4766,0.4457]

plt.plot(x,y1,label='viewpoint dataset',linewidth=2,color='c',marker='o',markerfacecolor='yellow',markersize=12)

plt.plot(x,y2,label='illumination dataset',linewidth=2,color='lime',marker='^',markerfacecolor='r',markersize=12)

plt.plot(x,y3,label='mean mAP on two datasets',linewidth=2,color='blueviolet',marker='p',markerfacecolor='crimson',markersize=12)

plt.ylim(0.0,0.82)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Different layers of architecture',fontsize=12)
plt.ylabel('mean Average Precision (mAP)',fontsize=12)
plt.legend(fontsize=12)
plt.show()
