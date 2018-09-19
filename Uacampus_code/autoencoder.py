from keras.layers import Input, Dense  
from keras.models import Model  
import numpy as np  
def autoencoder(des):
    encoding_dim =256
    input_img = Input(shape=(des.shape[1],))  

    encoded = Dense(1024, activation='relu')(input_img)  
    encoded = Dense(512, activation='relu')(encoded)  
    decoded_input = Dense(256, activation='relu')(encoded)  
  
    decoded = Dense(512, activation='relu')(decoded_input)  
    decoded = Dense(1024, activation='relu')(decoded)  
    decoded = Dense(des.shape[1], activation='relu')(encoded)  
  
    autoencoder = Model(inputs=input_img, outputs=decoded)  
    encoder = Model(inputs=input_img, outputs=decoded_input)  
  
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')  
  
 
    autoencoder.fit(des,des, epochs=20, batch_size=16,shuffle=False)  

    encoded_imgs = encoder.predict(des)  
    print(encoded_imgs.shape)
    return encoded_imgs
