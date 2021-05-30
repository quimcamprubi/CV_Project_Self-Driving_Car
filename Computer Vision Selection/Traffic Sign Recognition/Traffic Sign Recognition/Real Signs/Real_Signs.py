import cv2
import numpy as np

from os import listdir
from os.path import isfile, join                     
import pandas as pd                        
import matplotlib.pyplot as plt                  
import cv2             
import tensorflow as tf                                                   
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
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image


def model_signs():    
    data = []
    labels = []
    classes = 43
    for i in range(classes):
        path = os.path.join(os.getcwd(),'signs_dataset\\Train',str(i))
        images = os.listdir(path)
        
        for j in images:
            try:
                image = Image.open(path + '\\'+ j)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
                temp_vect = np.zeros(43)
                temp_vect[i] = 1
                labels.append(temp_vect)
            except:
                print("Error loading image")
    #Converting lists into numpy arrays bcoz its faster and takes lesser #memory
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape, labels.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=68)
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(43, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    model.save("Trafic_signs_model.h5")
    #plotting graphs for accuracy 
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #plotting graphs for loss 
    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def test_model(model):
    y_test = pd.read_csv('signs_dataset/Test.csv')
    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open('signs_dataset/'+img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    pred = model.predict_classes(X_test)
    #Accuracy with the test data
    print(accuracy_score(labels, pred))
    
    
def GUI(model):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }
           
    window = Tk()
    window.geometry('600x500')
    window.title('Traffic sign classifier')
    
    window.configure(background='#1e3e64')
    
    heading = Label(window, text="Traffic Sign Classifier",padx=220, font=('Verdana',20,'bold'))
    heading.configure(background='#143953',foreground='white')
    heading.pack()
    
    sign = Label(window)
    sign.configure(background='#1e3e64')
    
    value = Label(window,font=('Helvetica',15,'bold'))
    value.configure(background='#1e3e64')
    
    def classify(file_path):
        global label_packed
        image = Image.open(file_path)
        image = image.resize((30,30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        print(image.shape)
        pred = model.predict_classes([image])[0]
        sign = classes[pred+1]
        print(sign)
        value.configure(foreground='#ffffff', text=sign)
    
    def show_cb(file_path):
        classify_b=Button(window,text="Classify Image",command=lambda: classify(file_path),padx=20,pady=5)
        classify_b.configure(background='#147a81', foreground='white',font=('arial',10,'bold'))
        classify_b.place(relx=0.6,rely=0.80)
        
    def uploader():
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((window.winfo_width()/2.25),(window.winfo_height()/2.25)))
            im = ImageTk.PhotoImage(uploaded)
            
            sign.configure(image=im)
            sign.image=im
            value.configure(text='')
            show_cb(file_path)
        except:
            pass
    
    upload = Button(window,text="Upload an image",command=uploader,padx=10,pady=5)
    upload.configure(background='#e8d08e', foreground='#143953',font=('arial',10,'bold'))
    upload.pack()
    upload.place(x=100, y=400)
    
    sign.pack()
    sign.place(x=230,y=100)
    value.pack()
    value.place(x=240,y=300)
    
    window.mainloop()   
    
    
    
if __name__ == '__main__':
    '''
    model_signs()
    '''
    model = load_model('Trafic_signs_model.h5')
    
    test_model(model)
    
    GUI(model)
    

