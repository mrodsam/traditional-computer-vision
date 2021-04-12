#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:38:01 2019

@author: Marta Rodríguez Sampayo
"""

import cv2
import numpy as np
import math
import params as par
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('Modelo.png')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # normalizar la imagen al rango [0,1] de tipo float para facilitar las operaciones
    imageN = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #imageN = gray
    
    #Suavizar la imagen con un filtro Gaussiano de sigma = 0.5
    imageBlur = cv2.GaussianBlur(imageN, (0,0), par.INIT_SIGMA)
    
    #Doblar el tamaño de la imagen
    imageBlurResize = cv2.resize(imageBlur,None,fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
    
    pyramid = generate_gaussian_pyramid(imageBlurResize,4,par.INTVLS,par.SIGMA)
    
    dogpyramid = generate_dogpyramid(pyramid)
        
    candidates = detect_extrema(dogpyramid)
        
    extrema = []
    for cord in candidates:
        isExt, ext = refine_extrema(dogpyramid, pyramid, cord[0], cord[1], cord[2], cord[3])
        if(isExt):
            extrema.append(ext)
    
       
    keypoints = []
    for kp in extrema:
        angle = assign_orientation(pyramid, kp)
        kp.append(angle)
        keypoints.append(kp)
    
    print ('Puntos de interés encontrados: '+str(len(keypoints)))
        
    descriptor = []
    for kp in keypoints:
        iskp, ret = calculate_descriptor(pyramid, kp)
        if iskp:
            descriptor += ret
        

    plt.imshow(imageBlurResize, cmap='gray')

    i = np.array([])
    x = np.array([])
    y = np.array([])
    
    for index in range(len(keypoints)):
        i = np.append(i, keypoints[index][0])
        x = np.append(x, keypoints[index][2])
        y = np.append(y, keypoints[index][3])
    x = x * np.power(2, i)
    y = y * np.power(2, i)
    plt.scatter(x=y, y=x, c='r', s=10)
    plt.show()
    
def generate_octave(init_level, s, sigma):
    octave = [] #lista con las imágenes de un nivel de la pirámide
    k= 2**(1/par.INTVLS) #Factor que separa las sigmas de cada escala
    
    #Número de imágenes por octava
    for j in range (par.INTVLS+2):  
        next_level = cv2.GaussianBlur(init_level,(0,0),sigma * (k ** (j+1)))
        octave.append(next_level)
        init_level = octave[-1]
            
    return octave

def generate_gaussian_pyramid(im, num_octave, s, sigma): 
  pyr = [] 
  
  for _ in range(num_octave): 
    octave = generate_octave(im, s, sigma) 
    pyr.append(octave)
    #Re-escalado
    im = cv2.resize(octave[-3],None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
  return pyr

def generate_DoG_octave(gaussian_octave): 
  octave = [] 
  
  for i in range(1, len(gaussian_octave)):   
    octave.append(gaussian_octave[i] - gaussian_octave[i-1])
    
  return octave


def generate_dogpyramid(gaussian_pyramid): 
  pyr = [] 
  
  for gaussian_octave in gaussian_pyramid: 
    pyr.append(generate_DoG_octave(gaussian_octave)) 
    
  return pyr

def detect_extrema(dogpyramid):
    candidates = []
    localArea = []
    
    for octave in range(len(dogpyramid)):
        for scale in range(1, len(dogpyramid[octave])-1):
            prevImg = dogpyramid[octave][scale-1]
            currentImg = dogpyramid[octave][scale]
            nextimg = dogpyramid[octave][scale+1]
            
            coordinates = []
            for row in range(1, dogpyramid[octave][scale].shape[0]-1):
                for column in range(1, dogpyramid[octave][scale].shape[1]-1): 
                    
                    localArea.extend(prevImg[row-1:row + 2, column-1:column+2].flatten())
                    localArea.extend(currentImg[row - 1:row + 2, column - 1:column + 2].flatten())
                    localArea.extend(nextimg[row - 1:row + 2, column - 1:column + 2].flatten())
                    
                    currentPixel = currentImg[row, column]
                    
                    isMax = True
                    isMin = True
                    index = 0
                    
                    while(isMax or isMin) and index < len(localArea):
                        if index != 13:
                            if currentPixel <= localArea[index]:
                                isMax = False
                            if currentPixel >= localArea[index]:
                                isMin = False
                        index += 1
                        
                    if isMax or isMin:
                        coordinates.append([octave,scale,row, column])

                    localArea = []

        candidates+=coordinates
        
    return candidates



def refine_extrema(dogpyramid,pyramid,i,j,x,y):
    
    ret = []
    
    dx = (dogpyramid[i][j][x+1][y] - dogpyramid[i][j][x-1][y]) * 0.5
    dy = (dogpyramid[i][j][x][y+1] - dogpyramid[i][j][x][y-1]) * 0.5
    ds = (dogpyramid[i][j+1][x][y+1] - dogpyramid[i][j-1][x][y]) * 0.5
    
    dxx = dogpyramid[i][j][x+1][y] + dogpyramid[i][j][x-1][y] - 2 * dogpyramid[i][j][x][y]
    dxy = (dogpyramid[i][j][x+1][y+1] + dogpyramid[i][j][x-1][y-1] - dogpyramid[i][j][x+1][y-1]- dogpyramid[i][j][x-1][y+1]) * 0.25
    dxs = (dogpyramid[i][j+1][x+1][y] + dogpyramid[i][j-1][x-1][y] - dogpyramid[i][j-1][x+1][y] - dogpyramid[i][j+1][x-1][y]) * 0.25
    dyy = dogpyramid[i][j][x][y+1] + dogpyramid[i][j][x][y-1] - 2 * dogpyramid[i][j][x][y]
    dys = (dogpyramid[i][j+1][x][y+1] + dogpyramid[i][j-1][x][y-1] - dogpyramid[i][j-1][x][y+1] - dogpyramid[i][j+1][x][y-1]) * 0.25
    dss = dogpyramid[i][j+1][x][y] + dogpyramid[i][j-1][x][y] - 2 * dogpyramid[i][j][x][y]
    
    
    grad = np.array([dx, dy, ds])
    hessian = np.array([
            [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]
            ])
    
    #Refinar localización
    offset = -1.0 * np.dot(np.linalg.inv(hessian), grad)
    
    if math.fabs(offset[0]) > 0.5 or math.fabs(offset[1]) > 0.5 or math.fabs(offset[2]) > 0.5:  # Numeric stability check
        return False, ret
    
    #Bajo contraste: se rechazan los candidatos con ratio <0.03
    value = dogpyramid[i][j][x][y] + 0.5 * np.dot(grad, offset)
    
    if(np.abs(value) < par.CONTR_THR):
        return False, ret
    
    #Curvatura
    trace = hessian[0][0] + hessian[1][1]
    det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0]
    if(det <= 0):
        return False, ret
    
    if(trace * trace / det >= (par.CURV_THR + 1) * (par.CURV_THR + 1) / par.CURV_THR):
        return False, ret
    
    mag = pyramid[i][j][x][y]
    ret = [i, j + offset[2], x + offset[0], y + offset[1], value + mag]
    
    return True, ret

def gaussianKernel(ksize, sig):
    k = cv2.getGaussianKernel(ksize, sig)
    kern = np.outer(k, k)
    kern = kern * (1. / kern[int(round(ksize / 2))][int(round(ksize / 2))])
    return kern
    
def assign_orientation(pyramid,kp):
    
    div = 360 / par.NUM_BINS

    radius = 3
    window = 7
    k = 2**(1/par.INTVLS)
    sig = ((2 ** kp[0]) * (k ** kp[1]) * par.SIGMA) * 1.5

    point = np.rint(kp).astype(int)
    gkern = gaussianKernel(window, sig)
    histogram = np.zeros(par.NUM_BINS)
    

    #Se crea una lista que almacena gradientes de 2 dimensiones alrededor del punto de interés en una ventana 7x7
    grad_voxel_img = [0, 0]
    grad_voxel_img[0], grad_voxel_img[1] = np.gradient(
        pyramid[point[0]][point[1]][max(point[2] - radius, 0):point[2] + radius + 1,
        max(0, point[3] - radius):point[3] + radius + 1])
    
    #Se calcula la magnitud y el ángulo de dichos gradientes
    grad_mag = np.sqrt(np.square(grad_voxel_img[0]) + np.square(grad_voxel_img[1]))
    grad_angle = np.arctan2(grad_voxel_img[1], grad_voxel_img[0])
    
    grad_angle *= 180.0 / np.pi         # Radianes a grados
    grad_angle = grad_angle % 360.0     # [-180, 180] -> [0, 360]

    grad_angle = grad_angle * 0.1
    grad_angle = grad_angle.astype(int)

    for i in range(window):
        for j in range(window):
            histogram[grad_angle[i][j]] += grad_mag[i][j] * gkern[i][j]

    r = np.argmax(histogram)
    angle = []
    angle.append(r * div + div / 2.0)

    m = histogram[r]
    p = m * 0.8
    for i in range(len(histogram)):
        if (histogram[i] >= p and i != r):
            angle.append(i * div + div / 2.0)

    return angle

    
def calculate_descriptor(pyramid, kp):

  
    i = int(round(kp[0]))
    j = int(round(kp[1]))
    x = int(round(kp[2]))
    y = int(round(kp[3]))
    
    ang = kp[5]
    no_of_kp = len(ang)
    k = 2**(1/par.INTVLS)

    desc_list = []

    if (x < 8 or x > pyramid[i][j].shape[0] - 8 - 1):
        return False, desc_list
    if (y < 8 or y > pyramid[i][j].shape[1] - 8 - 1):
        return False, desc_list

    sig = ((2 ** kp[0]) * (k ** kp[1]) * par.SIGMA) * 0.5

    gkern = gaussianKernel(16, sig)
    desc_size = 8
    no_of_bins = 8
    bins = 360 / no_of_bins

    grad_voxel_img = [0, 0]
    grad_voxel_img[0], grad_voxel_img[1] = np.gradient(
        pyramid[i][j][x - desc_size:x + desc_size,
        y - desc_size:y + desc_size])
    grad_mag = np.sqrt(np.square(grad_voxel_img[0]) + np.square(grad_voxel_img[1]))
    grad_angle = np.arctan2(grad_voxel_img[1], grad_voxel_img[0])
    grad_angle *= 180.0 / np.pi  # Radianes a grados
    grad_angle = grad_angle % 360.0  # [-180, 180] -> [0, 360]

    threshold = 0.2

    for m in range(no_of_kp):
        desc = np.array([])
        for p in range(4):
            for q in range(4):
                hist = np.zeros(no_of_bins)
                for r in range(4):
                    for s in range(4):
                        a = (grad_angle[p * 4 + r][q * 4 + s] - ang[m]) / bins
                        t = np.int(np.floor(a))
                        hist[t] += gkern[p * 4 + r][q * 4 + s] * grad_mag[p * 4 + r][q * 4 + s]
                desc = np.append(desc, hist)
        desc.reshape((4, 4, 8))

        # Normalización
        norm = math.sqrt(np.sum(np.square(desc)))
        desc = desc / norm

        # Umbral para invarianza ante cambios de contraste
        desc = np.where(np.greater(desc, threshold), threshold, desc)

        # Se normaliza de nuevo
        norm = math.sqrt(np.sum(np.square(desc)))
        desc = desc / norm

        desc_list.append(desc)

    return True, desc_list

if __name__ == '__main__':
    main()

