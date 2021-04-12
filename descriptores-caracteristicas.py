#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:13:09 2020

@author: martarodriguezsampayo
"""

# SIFT, SURF,FAST, BRIEF, HOG y ORB
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from skimage.feature import hog
from skimage import exposure

MIN_MATCH_COUNT = 10
MATCHING = True #True:proceso completo, False:solo puntos clave
DATASET1PATH = 'dataset1_reyDistorsiones/'
DATASET2PATH = 'dataset2_reyCopia/reyCopia/'
dataset1 = False #True:dataset1, False:dataset2
save = False #True:guardar las figuras
detector = 1 #1:SIFT, 2:SURF, 3:FAST, 4:BRIEF, 5:HOG, 6:ORB, 7:BRISK, 8:AKAZE

def main():
    i = 0
    if dataset1:
        globPath = DATASET1PATH
    else:
        globPath = DATASET2PATH
        
    for filename in glob.glob(globPath+"*"):
        if not dataset1 or filename.startswith("dataset1_reyDistorsiones/t"):
           
            modelfile = 'dataset1_reyDistorsiones/Modelo.png'
            model = cv2.imread(modelfile)
            image = cv2.imread(filename)
            gray_model = cv2.cvtColor(model,cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
            binary, keypoints1, keypoints2, descriptors1, descriptors2 = select_detector(detector, gray_model, gray_image)
            
            #HoG: solo mostrar las imágenes
            if detector == 5:
                continue
            #FAST: solo mostrar puntos de interés detectados
            if MATCHING and detector != 3:
                if binary:
                    good_matches = calculate_matches_binary(descriptors1, descriptors2)
                else:
                    good_matches = calculate_matches(descriptors1, descriptors2)
                
                matchesMask, img2, img_transform = find_transformation(keypoints1, keypoints2, good_matches, gray_model, gray_image)
                
                draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 2)
                img3 = cv2.drawMatches(gray_model,keypoints1,img2,keypoints2,good_matches,None,**draw_params)
                plt.imshow(img3, 'gray')
                if save:
                    plt.savefig("matching"+str(filename.split("/")[2]))
                    i+=1
                plt.show()
                if img_transform is not None:
                    plt.imshow(img_transform, 'gray')
                    if save:
                        plt.savefig("transform-"+str(filename.split("/")[2]))
                    plt.show()
            else:
                image_keypoints = cv2.drawKeypoints(gray_image, keypoints2, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                plt.imshow(image_keypoints, 'gray')
                if save:
                    plt.savefig("keypoints-"+str(i))
                    i+=1
                plt.show()
         
def find_transformation(keypoints1, keypoints2, good_matches, img1, img2):
    
    if len(good_matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
       
        print('%d / %d  inliers/matched' % (np.sum(mask), len(mask)))
        matchesMask = mask.ravel().tolist()
        
        invM = np.linalg.inv(M)
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        h2,w2 = img2.shape
        transform = cv2.warpPerspective(img2, invM, (2*w,2*h))
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print( "No se han encontrado suficientes correspondencias - {}/{}".format(len(good_matches), MIN_MATCH_COUNT) )
        matchesMask = None
        transform = None
        
    return matchesMask, img2, transform

def calculate_matches_binary(descriptors1, descriptors2):
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1) 
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(descriptors1,descriptors2,k=2)
    
    # ratio test as per Lowe's paper
    ratio_thresh = 0.7
    good_matches = []
    for i,match in enumerate(matches):
        if not match or len(match) != 2:
            continue
        else:
            m,n = match
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
            
    return good_matches

def calculate_matches(descriptors1, descriptors2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(descriptors1,descriptors2,k=2)
    
    # ratio test as per Lowe's paper
    ratio_thresh = 0.7
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

            
    return good_matches


def select_detector(id, img1, img2):
    binary = False
    # SIFT
    if(id == 1):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1,None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2,None)
        
    # SURF
    if(id == 2):
        surf = cv2.xfeatures2d.SURF_create()
        keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
        keypoints2, descriptors2 = surf.detectAndCompute(img2, None)
        
    # FAST
    if(id == 3):
        fast = cv2.FastFeatureDetector_create()
        keypoints1 = fast.detect(img1, None)
        keypoints2 = fast.detect(img2, None)
        descriptors1 = None
        descriptors2 = None
        
    # BRIEF
    if(id == 4):
        fast = cv2.FastFeatureDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1 = fast.detect(img1, None)
        kp2 = fast.detect(img2, None)
        keypoints1, descriptors1 = brief.compute(img1, kp1)
        keypoints2, descriptors2 = brief.compute(img2, kp2)
        binary = True

    # HOG
    if(id == 5):
        
        fd, hog_image2 = hog(img2, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=False)
        
        hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 10))
        
        
        plt.imshow(hog_image_rescaled2, cmap=plt.cm.gray)
        if save:
            plt.savefig("hogImage2")
        plt.show()
        return None, None, None, None, None    
        
    # ORB
    if(id == 6):
        orb = cv2.ORB_create(nfeatures=1500)
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
        binary = True
    
    #BRISK
    if(id == 7):
        brisk = cv2.BRISK_create()
        keypoints1, descriptors1 = brisk.detectAndCompute(img1, None)
        keypoints2, descriptors2 = brisk.detectAndCompute(img2, None)
        binary = True
    
    #AKAZE
    if id == 8:
        akaze = cv2.AKAZE_create()
        keypoints1, descriptors1 = akaze.detectAndCompute(img1, None)
        keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)
        binary = True        
    
    return binary, keypoints1, keypoints2, descriptors1, descriptors2

if __name__ == '__main__':
    main()
