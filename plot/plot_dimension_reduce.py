import matplotlib.pyplot as plt
'''
x=['conv2','conv3','conv4','conv5']
y1=[0.2807,0.3485,0.3601,0.3734]
y2=[0.3504,0.4233,0.4311,0.4123]
y3=[0.3425,0.4030,0.4051,0.3897]
'''
x1=[30.976,43.264,46.464,64.896,120.000,139.968]
y1=[789/354,803/354,830/354,898/354,990/354,1274/354]

x2=[73.728,100.352,346.112,401.408,746.496,802.816]
y2=[1325/354,1338/354,2154/354,2398/354,3571/354,3711/354]

x3=[51.200,100.352,147.456,200.704,346.112,401.408,746.496,802.816]
y3=[1440/354,1441/354,1599/354,1778/354,2108/354,2213/354,3657/354,3993/354]

x4=[41.600,50.176,184.320,200.704,346.112,401.408,746.496,802.816]
y4=[1228/354,1333/354,1672/354,1741/354,1764/354,2125/354,3326/354,3576/354]

x5=[30.976,41.600,43.264,46.464,51.200,64.896,73.728,100.352,120.000,139.968,147.456,184.320,200.704,346.112,401.408,746.496,802.816]
y5=[789/354,1228/354,803/354,830/354,1440/354,896/354,1325/354,1338/354,990/354,1274/354,1599/354,1672/354,1741/354,1764/354,2125/354,3326/354,3576/354]

plt.plot(x4,y4,label='AlexNet',linewidth=2,color='c',marker='o',markerfacecolor='yellow',markersize=12)

#plt.plot(x2,y2,label='VGG16',linewidth=2,color='lime',marker='^',markerfacecolor='r',markersize=12)

#plt.plot(x3,y3,label='ResNet101',linewidth=2,color='blueviolet',marker='p',markerfacecolor='crimson',markersize=12)

#plt.plot(x4,y4,label='DenseNet169',linewidth=2,color='y',marker='*',markerfacecolor='b',markersize=12)


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Dimension of descriptors (*1000)',fontsize=12)
plt.ylabel('Matching time (s)',fontsize=12)
#plt.legend(fontsize=12)
plt.show()
