#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:47:03 2020

@author: martarodriguezsampayo
"""
import cv2
import glob
import numpy as np

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

features = 1 #0:contornos, 1:roi
classes = 3 #Número de clases(2:"enfermos" y "sanos"; 3:"enfermos_DCL_estables", "enfermos_DCL_evolucion" y "sanos")
modelName = 'casa' #bucles, casa, circulo, cruz, cuadrado, cubo, minimental, picosmesetas, triangulo

def main():
    if features == 0:
        print( "Modelo: {}, Número de clases: {}, Características: Contornos".format(modelName, classes) )
    else:
        print( "Modelo: {}, Número de clases: {}, Características: Regiones de interés".format(modelName, classes) )
    
    feature1, feature2, feature3, label = generate_dataset()
    knn(feature1, feature2, feature3, label)
        
def generate_dataset():
    feature1 = []
    feature2 = []
    feature3 = []
    label = []
    count1 = 0
    count2 = 0
    count3 = 0
    for idfolder in glob.glob("dataset3_DCL/dibujos/enfermos_DCL_estables/*"):
        for filename in glob.glob(idfolder+"/"+modelName+"*"):
            if features == 0:
                num, area, perimeter = findFeaturesContours(filename)
            else:
                num, area, perimeter = findFeaturesRoi(filename)
            feature1.append(num)
            feature2.append(area)
            feature3.append(perimeter)
            if classes == 3:
                label.append("enfermos_DCL_estables")
            else:
                label.append("enfermos") 
            count1+=1           
            
    for idfolder in glob.glob("dataset3_DCL/dibujos/enfermos_DCL_evolucion/*"):
        for filename in glob.glob(idfolder+"/"+modelName+"*"):
            if features == 0:
                num, area, perimeter = findFeaturesContours(filename)
            else:
                num, area, perimeter = findFeaturesRoi(filename)
            feature1.append(num)
            feature2.append(area)
            feature3.append(perimeter)
            if classes == 3:
                label.append("enfermos_DCL_evolucion")
            else:
                label.append("enfermos")
            count2+=1
            
    for idfolder in glob.glob("dataset3_DCL/dibujos/sanos/*"):
        for filename in glob.glob(idfolder+"/"+modelName+"*"):
            if features == 0:
                num, area, perimeter = findFeaturesContours(filename)
            else:
                num, area, perimeter = findFeaturesRoi(filename)
            feature1.append(num)
            feature2.append(area)
            feature3.append(perimeter)
            label.append("sanos")
            count3+=1
            
    print( "Muestras estables: {}, Muestras evolución: {}, Muestras sanos: {}".format(count1, count2, count3) )
                      
    return feature1, feature2, feature3, label
        
def findFeaturesContours(filename):
    image = cv2.imread(filename)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      
    edged = cv2.Canny(gray, 30, 200) 
      
    _, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
          
    area = 0
    per = 0
    for cnt in contours:
        area += cv2.contourArea(cnt)
        per += cv2.arcLength(cnt, False)
    
    return len(contours), str(int(area)), str(int(per))

def findFeaturesRoi(filename):
    image = cv2.imread(filename) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # dilation
    kernel = np.ones((50, 60), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    # find contours
    # cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) >= 4:
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    bbox_area = []
    bbox_per = []
    pos_bboxes = []
    for i, ctr in enumerate(sorted_ctrs):
        box = [0] *4
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        bbox_area.append(w*h)
        bbox_per.append(2*w+2*h)
    
        box[0] = x
        box[1] = y
        box[2] = x + w
        box[3] = y + h
        pos_bboxes.append(box)
        
    if not bbox_area:
        return 0, 0, 0
    
    bbox_area = np.array(bbox_area)
    max_area = np.argmax(bbox_area, axis=0)
    max_per = np.argmax(bbox_per, axis=0)

    
    return len(bbox_area), bbox_area[max_area], bbox_per[max_per]

def knn(feature1, feature2, feature3, label):

    #creating labelEncoder
    le = preprocessing.LabelEncoder()
  
    feature1_encoded = le.fit_transform(feature1)
    feature2_encoded = le.fit_transform(feature2)
    feature3_encoded = le.fit_transform(feature3)
    nlabel = le.fit_transform(label)
    
    #combinig features into single list of tuples
    features = list(zip(feature1_encoded, feature2_encoded, feature3_encoded))
    
    X_train, X_test, y_train, y_test = train_test_split(features, nlabel, test_size=0.2)
    print( "Muestras entrenamiento: {}, Muestras validación: {}".format(len(X_train), len(X_test)) )
    
    if features == 0:
        model = KNeighborsClassifier(n_neighbors=21)
    else:
        model = KNeighborsClassifier(n_neighbors=15)

    evaluation(model, X_train, y_train, X_test, y_test)
    

def evaluation(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(metrics.classification_report(y_test, y_pred))
    
    return acc
if __name__ == '__main__':
    main()