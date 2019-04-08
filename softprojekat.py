# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:56:43 2019

@author: djordjije
"""
import numpy as np
import matplotlib
import cv2 # OpenCV
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16,12

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    #image_binary = np.ndarray((height, width), dtype=np.uint8)
    #63 83%
    ret,image_bin = cv2.threshold(image_gs, 206, 255, cv2.THRESH_BINARY)
    return image_bin
def h_transformation(pic):
    return cv2.HoughLinesP(pic,1,np.pi/180,60,50,50)

def display_image(image, color= False):
    if color:
        #plt.figure()
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
        
def erode(image, kernel):   
    return cv2.erode(image, kernel, iterations=1)


def find_line(lines): 
    
    x1 = min(lines[:,0,0])
    y1 = max(lines[:,0,1])
    x2 = max(lines[:,0,2])
    y2 = min(lines[:,0,3])
    
    line = [x1, y1, x2, y2]
    return line
def distance(x1,x2,y1,y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

lastX = 100000
lastY = 1000000
def select_roi(image_orig, image_bin, line):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    array_x = []
    array_y = []
    array_xy = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        global lastX
        global lastY
        
        #lastX = x
        #lastY = y
        area = cv2.contourArea(contour)
        dist=15
        dist1 = distance(x1,x2,y1,y2)
        dist2 = distance(x1,x-dist, y1, y-dist) + distance(x2, x-dist, y2, y-dist)
        
        diff = 0.09
        
        if abs(dist1 - dist2) < diff:
            
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
             if (((w > 1 and h > 10) or (w>14 and h>7)) and (w<=28 and h<=28)):
                 k = 5
                 #print([x,y])
                 array_x.append(x)
                 array_y.append(y)
                 array_xy.append([x,y])
                 region = image_bin[y-k:y+h++1+k ,x-k:x+w+1+k]
                 regions_array.append([cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST), (x,y,w,h)])       
                 cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, array_xy
def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann
def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]
def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result    
              
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
ann = model_from_json(model_json)
ann.load_weights("model.h5")    
    

file= open("out.txt","w+")
file.write("RA 111/2015 Djordjije Ivanisevic\r")
file.write("file	sum\r")

for i in range(0,10) :
    cap = cv2.VideoCapture('data/video-'+str(i)+'.avi')
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        
    
    frame_num = 0
    cap.set(1, frame_num) 
        
    ret_val, frame = cap.read()
        
        
    kernel = np.ones((3,3))
    pic1 = frame[:,:,0]
       
    lines = h_transformation(erode(pic1, kernel))
    
    plus_line = find_line(lines)
    #print(lines)
    #print(plus_line[0], plus_line[1], plus_line[2], plus_line[3])
    
    pic2 = frame[:,:,1]
    lines2 = h_transformation(erode(pic2, kernel))
    
    minus_line = find_line(lines2)
    #print(lines)
    #print(minus_line[0], minus_line[1], minus_line[2], minus_line[3])
    
    
    pregions = []
    mregions = []
    
    pr_final = []
    mr_final = []
    
    xy_final1 = []
    xy_final2 = []
    
    img_plot1 = []
    img_plot2 = []
     
    while(cap.isOpened()):
          # Capture frame-by-frame
          ret, frame = cap.read()
          frame_num += 1
          
          
          if ret == True:
              binaryPic = image_bin(image_gray(frame))
              
              image_orig1, plus_regions, array_xy1 = select_roi(frame, binaryPic, plus_line)
              image_orig2, minus_regions, array_xy2 = select_roi(frame, binaryPic, minus_line)
              
              
              for m in range (0, len(plus_regions)):
                     
                  pregions.append(plus_regions[m])
                  xy_final1.append(array_xy1[m])    
                  img_plot1.append(image_orig1)
                      
                  #display_image(image_orig1)
                  #plt.figure()
              for n in range(0, len(minus_regions)):
                  mregions.append(minus_regions[n])
                  xy_final2.append(array_xy2[n])
                  img_plot2.append(image_orig2)
                  #display_image(image_orig2)
                  #plt.figure()
                  #display_image(image_orig2)
                  #plt.figure()    
              #if frame_num==500:
               #   break
              #cv2.imshow('Detekcija brojeva',image_orig)
          else:
              break
    flag = True     
    #print(xy_final1[1][0])
    for mm in range (0, len(pregions)):
          
          if mm == 0:
              pr_final.append(pregions[mm])
              flag = True
              continue
          if distance(xy_final1[mm][0], xy_final1[mm-1][0], xy_final1[mm][1], xy_final1[mm-1][1]) < 2.5:
              if flag == True:
                  pr_final.append(pregions[mm])
                  flag = False
                  #display_image(pregions[mm])
                  #plt.figure() 
          else:
              flag = True
    flag2 = True     
    #print(len(pr_final), len(pregions))
    for nn in range (0, len(mregions)):
          
          if nn == 0:
              mr_final.append(mregions[nn])
              flag2 = True
              continue
          if distance(xy_final2[nn][0], xy_final2[nn-1][0], xy_final2[nn][1], xy_final2[nn-1][1]) < 2.5:
              if flag2 == True:
                  mr_final.append(mregions[nn])
                  flag2 = False
                 # display_image(mregions[nn])
                 # plt.figure() 
          else:
              flag2 = True      
                  
        
                  
                  
         
          
    alphabet = [0,1,2,3,4,5,6,7,8,9]
    if not(not pr_final):
        res_plus = ann.predict(np.array(prepare_for_ann(pr_final),np.float32))
        array_plus=display_result(res_plus,alphabet)
    if mr_final:
        res_minus = ann.predict(np.array(prepare_for_ann(mr_final),np.float32))
        array_minus=display_result(res_minus,alphabet)
    
    total_sum = sum(array_plus) - sum(array_minus)
    
    #print(array_plus)
    #print(array_minus)
    
    
    print(total_sum)
    file.write('video-'+str(i)+'.avi\t' + str(total_sum)+'\r')
     
       
      
        
    

    






