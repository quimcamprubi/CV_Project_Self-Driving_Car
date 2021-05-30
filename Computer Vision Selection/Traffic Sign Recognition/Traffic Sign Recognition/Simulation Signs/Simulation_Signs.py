import cv2
import numpy as np

from os import listdir
from os.path import isfile, join                     
import pandas as pd                        
import matplotlib.pyplot as plt                  
import cv2             
import tensorflow as tf                          
from PIL import Image                           
import os                                        
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical          
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import tqdm                                    
import warnings
from sklearn.metrics import accuracy_score
import skimage.measure
import pytesseract


def im2double(im):
    out = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return out

'''
def get_dataset(crk3, clientID, counter_i):
    retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    #retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_buffer)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    while not resolution:
        retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    
    frame=np.array(frame,dtype=np.uint8)
    frame.resize([resolution[1],resolution[0],3])
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 1)
    plt.figure(1)
    plt.imshow(frame)
    plt.show()
    plt.imsave('claw_rover_k3/movement/signals_dataset/4/image1 ('+str(counter_i)+').png', frame)
    counter_i += 1
    return counter_i
'''

def cut_signal(im):
    minimi = 256
    maximi = 0
    minimj = 256
    maximj = 0
    dim0 = im[:,:,0] > 0.4
    dim1 = im[:,:,1] < 0.20
    dim2 = im[:,:,2] < 0.20
    dim_t = dim0 * dim1 * dim2
    imf = dim_t.astype('uint8')
    
    labeled_image = skimage.measure.label(imf, connectivity=2, return_num=True)
    
    counter = np.histogram(labeled_image[0], labeled_image[1]+1)[0]
    
    maxim = 0
    idmaxim = -1  
    for i in range(1, labeled_image[1]+1):
        if maxim < counter[i]:
            maxim = counter[i]
            idmaxim = i
        
    trobat1 = False
    trobat2 = False
    trobat3 = False
    trobat4 = False
    for i in range(256):
        for j in range(256):
            if (labeled_image[0][i][j] == idmaxim):
                if i < minimi: minimi = i; trobat1 = True
                if i > maximi: maximi = i; trobat2 = True
                if j < minimj: minimj = j; trobat3 = True
                if j > maximj: maximj = j; trobat4 = True  
    
    if trobat1 and trobat2 and trobat3 and trobat4:
        signal = im[minimi:maximi,minimj:maximj,:]
    else:
        signal = 0     
    return signal

def model_detect_signals():
    data = []
    labels = []
    classes = 4
    
    for i in range(0,classes):
        '''llegir imatges'''
        path = os.path.join(os.getcwd(),'signs_dataset\\'+str(i+1),'retalls')
        images = os.listdir(path)
        
        for j in images:
            try:
                image = Image.open(path + '\\'+ j).convert('RGB')
                image = np.array(image)
                image = cv2.resize(image, dsize=(30, 30), interpolation=cv2.INTER_CUBIC)
                image = image/255
                data.append(image)
                temp_vect = [0,0,0,0]
                temp_vect[i] = 1
                labels.append(temp_vect)
            except:
                print("Error loading image")
                
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape, labels.shape)
    
    
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=68)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=68)
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.05))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.05))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.05))
    model.add(Dense(4, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=8, epochs=25, validation_data=(X_val, y_val))
    model.save("Trafic_signs_model.h5")
    #plotting graphs for accuracy 
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    #plotting graphs for loss 
    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    
    
    score = model.evaluate(X_test,y_test,verbose=0)
    print('Test Score = ',score[0])
    print('Test Accuracy =', score[1])
 
    
def detect_signals(image, model):
    image = cut_signal(image)
    retorn = -1
    if type(image) is not int:
        if image.shape[0] > 40 and image.shape[1] > 40:
            image = cv2.resize(image, dsize=(30, 30), interpolation=cv2.INTER_CUBIC) 
            image = np.expand_dims(image, axis=0)
            pred = model.predict([image])[0]
            idpred = np.argmax(pred)
            if pred[idpred] > 0.7:
                if idpred == 0:
                    retorn = 'stop'
                elif idpred == 1:
                    retorn = '30'
                elif idpred == 2:
                    retorn = '50'
                elif idpred == 3:
                    retorn = '80'

    return retorn

if __name__ == '__main__':
    
    list_n=[19,43,28,50]
    for k in range(1,5):
        n_signs = list_n[k-1]
        for i in range(1,n_signs):
            imatge = plt.imread('signs_dataset/'+str(k)+'/imatge'+str(k)+' ('+str(i)+').png')
            imatge = cut_signal(imatge)
            if type(imatge) != int:
                plt.imsave('signs_dataset/'+str(k)+'/retalls/imatge'+str(k)+'_'+str(i)+'.png', imatge)
            
    model_detect_signals()
    
    model = load_model('Trafic_signs_model.h5')
    
    
    frameb = plt.imread('signs_dataset/2/imatge2 (25).png')[:,:,:3]
    frameb = im2double(frameb)
    print(detect_signals(frameb, model))
    
    




