from operator import itemgetter
import time
import os
from os import listdir
import sys
import numpy as np
import cv2
import pandas as pd

import keras
from keras import backend as K
K.set_image_dim_ordering("th")
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.models import model_from_json
from scipy.misc import imread,imresize
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

start_time=time.time()
num_channel=1

json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")

#load the pretrained model and compile it
loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

imname=sys.argv[1]
image = cv2.imread(imname, 1)
r1 = 750.0 / image.shape[0]
dim = (750, int(image.shape[0] * r1))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

imgcopy3 = resized.copy()
imgcopy2 = resized.copy()
imgcopy4 = resized.copy()
imgcopy5 = resized.copy()
imgcopy6 = resized.copy()
imgcopy7 = resized.copy()
imgcopy8 = resized.copy()
imgcopy9 = resized.copy()
imgcopy10 = resized.copy()
imgcopy = resized.copy()

blur = cv2.bilateralFilter(resized,9,35,35)
#cv2.imshow('blur', blur)
count=0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

def auto_canny(image, sigma):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def straws( blurd , chan ):
    u = 0
    edges = auto_canny(blurd, 0.33)
    #cv2.imshow('edges'+chan, edges)  
    
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph,kernel,iterations = 1)
    im, contours, hierarchy = cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    r = []
    arealist = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        arealist.append(area)
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        r.append(int(radius))

    d = {'col1':r, 'col2':arealist}
    df = pd.DataFrame(d)
    x = df.describe()
    a = x.loc['75%']
    rmax = a.iloc[0]
    amax = a.iloc[1]
    print (str(rmax),' ',str(amax))
    
    new_cont = []
    centr1 = []
    false = []
    true = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area >= amax - (amax/1.1) and area <= amax + (amax/1.1) ):
            (x,y),radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x),int(y))
            radius = int(radius)
            M = cv2.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            (x,y),(MA,ma),angle = cv2.fitEllipse(contours[i])
            ratio = ma/MA

            if (ratio >= 0.6):
                if (radius >= rmax-3 and radius <= rmax+3):
                    u = u+1
                    cnts = (contours[i],u)
                    true.append(cnts)
                    cv2.circle(imgcopy,center,radius,(255,0,0),2)
                    center = (int(x),int(y),radius,u)
                    centr1.append(center)
                
        false.append(contours[i])
    '''      
    for i in range(len(false)):
        rect = cv2.minAreaRect(false[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        
        if(area>= 1000 and area <= 5000):
            cv2.drawContours(imgcopy4,[box],-1,(0,0,255),2)
    #cv2.imshow("false from iter1",imgcopy4)
    '''
    morph = cv2.dilate(morph,kernel,iterations = 1)
    im, contours, hierarchy = cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    r = []
    arealist = []
    centr2 = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        arealist.append(area)
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        r.append(int(radius))

    d = {'col1':r, 'col2':arealist}
    df = pd.DataFrame(d)
    x = df.describe()
    a = x.loc['75%']
    rmax = a.iloc[0]
    amax = a.iloc[1]
    print (str(rmax),' ',str(amax))
    new_cont = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area >= amax - (amax/1.1) and area <= amax + (amax/1.1) ):
            (x,y),radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x),int(y))
            radius = int(radius)
            (x,y),(MA,ma),angle = cv2.fitEllipse(contours[i])
            ratio = ma/MA
            if (ratio >= 0.6):
                if (radius >= rmax-3 and radius <= rmax+3):
                    u = u+1
                    cnts = (contours[i],u)
                    true.append(cnts)
                    cv2.circle(imgcopy,center,radius,(0,255,0),2)
                    center = (int(x),int(y),radius,u)
                    centr2.append(center)
                
        false.append(contours[i])
        
    '''
    for i in range(len(false)):
        
        rect = cv2.minAreaRect(false[i])
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if(area>= 1000 and area <= 5000):
            cv2.drawContours(imgcopy5,[box],-1,(0,255,0),2)
    
    cv2.imshow("false from iter 2",imgcopy5)
    '''
    
    morph = cv2.dilate(morph,kernel,iterations = 1)
    im, contours, hierarchy = cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    r = []
    arealist = []
    centr3 = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        arealist.append(area)
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        r.append(int(radius))
    
    d = {'col1':r, 'col2':arealist}
    df = pd.DataFrame(d)
    x = df.describe()
    a = x.loc['75%']
    rmax = a.iloc[0]
    amax = a.iloc[1]
    print (str(rmax),' ',str(amax))
    new_cont = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area >= amax - (amax/1.1) and area <= amax + (amax/1.1) ):
            (x,y),radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x),int(y))
            radius = int(radius)
            (x,y),(MA,ma),angle = cv2.fitEllipse(contours[i])
            ratio = ma/MA
            if (ratio >= 0.6):
                if (radius >= rmax-3 and radius <= rmax+3 ):
                    u = u+1
                    cnts = (contours[i],u)
                    true.append(cnts)
                    cv2.circle(imgcopy,center,radius,(0,0,255),2)
                    center = (int(x),int(y),radius,u)
                    centr3.append(center)
              
        false.append(contours[i])
    '''    
    for i in range(len(false)):
        
        rect = cv2.minAreaRect(false[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if(area>= 1000 and area <= 5000):
            cv2.drawContours(imgcopy6,[box],-1,(255,0,0),2)

    cv2.imshow("fasle from iter 3",imgcopy6)
    '''
    morph = cv2.dilate(morph,kernel,iterations = 1)
    im, contours, hierarchy = cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    r = []
    arealist = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        arealist.append(area)
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        r.append(int(radius))
    d = {'col1':r, 'col2':arealist}
    df = pd.DataFrame(d)
    x = df.describe()
    a = x.loc['75%']
    rmax = a.iloc[0]
    amax = a.iloc[1]
    print (str(rmax),' ',str(amax))
    new_cont = []
    centr4 = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area >= amax - (amax/1.1) and area <= 6*amax):
            (x,y),radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x),int(y))
            radius = int(radius)
            (x,y),(MA,ma),angle = cv2.fitEllipse(contours[i])
            ratio = ma/MA
            if (ratio >= 0.6):
                if (radius >= rmax-3 and radius <= rmax+ 8):
                    u = u+1
                    cnts = (contours[i],u)
                    true.append(cnts)
                    cv2.circle(imgcopy,center,radius,(255,255,255),2)
                    center = (int(x),int(y),radius,u)
                    centr4.append(center)
               
        false.append(contours[i])
    '''    
    for i in range(len(false)):
        
        rect = cv2.minAreaRect(false[i])
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if(area>= 1000 and area <= 5000):
            cv2.drawContours(imgcopy7,[box],-1,(255,255,255),2)
    cv2.imshow("false from iter 4",imgcopy7)
    '''

    dist = 0
    for x4,y4,r4,u4 in centr4:
        for x3,y3,r3,u3 in centr3:
            dist = np.sqrt((y4-y3)**2 + (x4-x3)**2)
            if(dist < 10):
                centr3.remove((x3,y3,r3,u3))
        for x2,y2,r2,u2 in centr2:
            dist = np.sqrt((y4-y2)**2 + (x4-x2)**2)
            if(dist < 10):
                centr2.remove((x2,y2,r2,u2))
        for x1,y1,r1,u1 in centr1:
            dist = np.sqrt((y4-y1)**2 + (x4-x1)**2)
            if(dist < 10):
                centr1.remove((x1,y1,r1,u1))
        
        
    for x3,y3,r3,u3 in centr3:
        for x2,y2,r2,u2 in centr2:
            dist = np.sqrt((y3-y2)**2 + (x3-x2)**2)
            if(dist < 10):
                centr2.remove((x2,y2,r2,u2))
        for x1,y1,r1,u1 in centr1:
            dist = np.sqrt((y3-y1)**2 + (x3-x1)**2)
            if(dist < 10):
                centr1.remove((x1,y1,r1,u1))
        
    for x2,y2,r2,u2 in centr2:
        for x1,y1,r1,u1 in centr1:
            dist = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            if(dist < 10):
                centr1.remove((x1,y1,r1,u1))

    l = len(centr1)    
    for i in range(l):
        j = i+1
        while(j < l):
            
            dist = np.sqrt((centr1[i][0]-centr1[j][0])**2 + (centr1[i][1]-centr1[j][1])**2 )
            if(dist < 20):
                for k in range(j,l-1):
                    centr1[k] = centr1[k+1]
                    
                centr1.remove(centr1[l-1])
                j = j-1
                l = l-1
            j = j+1

    fcenter = []
    fcenter.extend(centr4)
    fcenter.extend(centr3)
    fcenter.extend(centr2)
    fcenter.extend(centr1)
    final = []
    for i in range(len(centr4)):
        x = centr4[i][3]
        for j in range(len(true)):
            if(true[j][1] == x):
                final.append(true[j][0])
    for i in range(len(centr3)):
        x = centr3[i][3]
        for j in range(len(true)):
            if(true[j][1] == x):
                final.append(true[j][0])
    for i in range(len(centr2)):
        x = centr2[i][3]
        for j in range(len(true)):
            if(true[j][1] == x):
                final.append(true[j][0])
    for i in range(len(centr1)):
        x = centr1[i][3]
        for j in range(len(true)):
            if(true[j][1] == x):
                final.append(true[j][0])
    
    for x,y,r,u in centr1:
        cv2.circle(imgcopy3,(x,y),r,(255,0,0),2)
    for x1,y1,r1,u1 in fcenter:
        cv2.circle(imgcopy2,(x1,y1),r1,(255,255,255),2)
    
    finalist=[]   
    global count1
    count1=len(fcenter)
    global count2
    count2=0
    for i in range(len(final)):
        cv2.drawContours(imgcopy8,final,i,(255,255,255),2)

        x,y,w,h = cv2.boundingRect(final[i])
        
        blackin = np.zeros(resized.shape,np.uint8)
        cv2.drawContours(blackin,final,i,(255,255,255), 2 )
        roi = blackin[-40+y:y+80,-40+x:x+80]
        cv2.imwrite("contour"+str(i)+".jpg",roi)
        
        try:
            test_image = imread("contour"+str(i)+".jpg",flatten=True)
            test_image=cv2.resize(test_image,(120,120))
            test_image = np.array(test_image) 
            test_image = test_image.astype('float32')
            test_image /= 255
           
            if num_channel==1:
                if K.image_dim_ordering()=='th':
                    test_image= np.expand_dims(test_image, axis=0)
                    test_image= np.expand_dims(test_image, axis=0)
                    #print ("in theano",test_image.shape)
                else:
                    test_image= np.expand_dims(test_image, axis=3) 
                    test_image= np.expand_dims(test_image, axis=0)
                    #print (test_image.shape)

        #print out the names of corrupt files
        except (IOError, SyntaxError) as e:
            print("Bad file: "+"contour"+str(i)+".jpg" ) 
        
        #print((loaded_model.predict(test_image)))
        #print(loaded_model.predict_classes(test_image))

        #only draw countours if predicted as straw
        if loaded_model.predict_classes(test_image)==1:
            cv2.drawContours(imgcopy10,final,i,(255,255,255),2)
            count2=count2+1
            

    cv2.imwrite("usingImageProcessing.jpg", imgcopy2)
    cv2.imwrite("usingCustomCNN.jpg", imgcopy10)
    
if __name__=="__main__":
    straws(blur, '35')
    print("Straws counted using simple image processing: ", count1)    
    print("Straws counted using custom classifier: ", count2)
    print("Computation time --- %s seconds ---" %(time.time() - start_time))
    cv2.waitKey(0)


