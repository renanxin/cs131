import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from skimage.feature import hog
from skimage import data,color,exposure
from skimage.transform import rescale, resize, downscale_local_mean
import glob, os
import fnmatch
import time
import math
from Homeworks.homework7.util import *
from Homeworks.homework7.detection import *

########################################################################################################################(Part 1: Hog Representation)

image_paths=fnmatch.filter(os.listdir('face'),'*.jpg')      #filter函数实现是列表特殊字符的过滤和筛选
list.sort(image_paths)
n=len(image_paths)
face_shape=io.imread('face/'+image_paths[0],as_gray=True).shape

avg_face=np.zeros(face_shape)
for i,image_path in  enumerate(image_paths):
    image=io.imread('face/'+image_path,as_gray=True)
    avg_face=np.asarray(avg_face)+np.asarray(image)
avg_face=avg_face/n
(face_feature,face_hog)=hog_feature(avg_face)

'''
plt.figure()
plt.subplot(1,2,1)
plt.imshow(avg_face,cmap='gray')
plt.title('average face image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(face_hog,cmap='gray')
plt.axis('off')
plt.title('hog representation of face')
'''
########################################################################################################################(Part 2: Sliding Window)

#plt.show()

image_path = 'image_0001.jpg'

image = io.imread(image_path, as_grey=True)
image = rescale(image, 1)

(hogFeature, hogImage) = hog_feature(image)

(winH, winW) = face_shape
(score, r, c, response_map) = sliding_window(image, face_feature, stepSize=30, windowSize=face_shape)
crop = image[r:r+winH, c:c+winW]

#fig,ax = plt.subplots(1)
#ax.imshow(image,cmap='gray')
r#ect = patches.Rectangle((c,r),winW,winH,linewidth=1,edgecolor='r',facecolor='none')
#ax.add_patch(rect)
#plt.show()

#plt.imshow(response_map,cmap='viridis', interpolation='nearest')
#plt.title('sliding window')
#plt.show()

########################################################################################################################(Part 3: Image Pyramids)
'''3.1 Image Pyramid'''
'''
image_path='image_0001.jpg'

image=io.imread(image_path,as_gray=True)
image=rescale(image,1.2)
sum_r=0
sum_c=0

images=pyramid(image,scale=0.9)
for i,result in enumerate(images):
    (current_scale,img)=result
    if i==0:
        sum_c=img.shape[1]
    sum_r+=img.shape[0]
composite_image=np.zeros((sum_r,sum_c))
pointer=0
for i,result in enumerate(images):
    (current_scale,img)=result
    composite_image[pointer:pointer+img.shape[0],0:img.shape[1]]=img
    pointer+=img.shape[0]

plt.imshow(composite_image,cmap='gray')
plt.axis('off')
plt.title('image pyramid')
plt.show()
'''
'''3.2 Pyramid Score'''
'''
image_path='image_0338.jpg'
image = io.imread(image_path, as_grey=True)
image = rescale(image, 1.2)
(winH,winW)=face_shape
maxscore,maxr,maxc,max_scale,max_response_map=pyramid_score(image,face_feature,face_shape,stepSize=30,scale=0.8)
fig,ax = plt.subplots(1)
ax.imshow(rescale(image, max_scale),cmap='gray')
rect = patches.Rectangle((maxc,maxr),winW,winH,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()

plt.imshow(max_response_map, cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.show()
'''
########################################################################################################################(Deformable Parts Detection)
image_paths=fnmatch.filter(os.listdir('face'),'*.jpg')
parts=read_facial_labels(image_paths)
lefteyes, righteyes, noses, mouths = parts

#Typical shape for left eye
lefteye_h=10
lefteye_w=20

lefteye_shape=(lefteye_h,lefteye_w)
avg_lefteye=get_detector(lefteye_h,lefteye_w,lefteyes,image_paths)
(lefteye_feature,lefteye_hog)=hog_feature(avg_lefteye,pixel_per_cell=2)
'''
plt.subplot(1,3,1)
plt.imshow(avg_lefteye,cmap='gray')
plt.axis('off')
plt.title('average left eye image')

plt.subplot(1,3,2)
plt.imshow(lefteye_hog,cmap='gray')
plt.axis('off')
plt.title('average hog image')
plt.show()
'''
righteye_h = 10
righteye_w = 20

righteye_shape = (righteye_h, righteye_w)

avg_righteye = get_detector(righteye_h, righteye_w, righteyes, image_paths)

(righteye_feature, righteye_hog) = hog_feature(avg_righteye, pixel_per_cell=2)
'''
plt.subplot(1,3,1)
plt.imshow(avg_righteye,cmap='gray')
plt.axis('off')
plt.title('average right eye image')

plt.subplot(1,3,2)
plt.imshow(righteye_hog,cmap='gray')
plt.axis('off')
plt.title('average hog image')
plt.show()
'''
nose_h = 30
nose_w = 26

nose_shape = (nose_h, nose_w)

avg_nose = get_detector(nose_h, nose_w, noses, image_paths)

(nose_feature, nose_hog) = hog_feature(avg_nose, pixel_per_cell=2)
'''
plt.subplot(1,3,1)
plt.imshow(avg_nose,cmap='gray')
plt.axis('off')
plt.title('average nose image')

plt.subplot(1,3,2)
plt.imshow(nose_hog,cmap='gray')
plt.axis('off')
plt.title('average hog image')
plt.show()
'''

mouth_h = 20
mouth_w = 36

mouth_shape = (mouth_h, mouth_w)

avg_mouth = get_detector(mouth_h, mouth_w, mouths, image_paths)

(mouth_feature, mouth_hog) = hog_feature(avg_mouth, pixel_per_cell=2)

detectors_list = [lefteye_feature, righteye_feature, nose_feature, mouth_feature]
'''
plt.subplot(1,3,1)
plt.imshow(avg_mouth,cmap='gray')
plt.axis('off')
plt.title('average mouth image')

plt.subplot(1,3,2)
plt.imshow(mouth_hog,cmap='gray')
plt.axis('off')
plt.title('average hog image')
plt.show()
'''

########################################################################################################################(Human Parts Location)
# test for computer_displacement
'''
test_array = np.array([[0,1],[1,2],[2,3],[3,4]])
test_shape=(6,6)
mu,std=compute_displacement(test_array,test_shape)
assert(np.all(mu == [1,0]))
assert(np.sum(std-[ 1.11803399,  1.11803399])<1e-5)
print("Your implementation is correct!")
'''
lefteye_mu, lefteye_std = compute_displacement(lefteyes, face_shape)
righteye_mu, righteye_std = compute_displacement(righteyes, face_shape)
nose_mu, nose_std = compute_displacement(noses, face_shape)
mouth_mu, mouth_std = compute_displacement(mouths, face_shape)

image_path = 'image_0338.jpg'
image = io.imread(image_path, as_grey=True)
image = rescale(image, 1.0)

(face_H, face_W) = face_shape
max_score, face_r, face_c, face_scale, face_response_map = pyramid_score(image, face_feature, face_shape,stepSize = 30, scale=0.8)

#plt.imshow(face_response_map,cmap='viridis', interpolation='nearest')
#plt.axis('off')
#plt.show()
max_score, lefteye_r, lefteye_c, lefteye_scale, lefteye_response_map = pyramid_score(image, lefteye_feature,lefteye_shape, stepSize = 20,scale=0.9, pixel_per_cell = 2)

lefteye_response_map = resize(lefteye_response_map, face_response_map.shape)
#plt.imshow(lefteye_response_map,cmap='viridis',interpolation='nearest')
#plt.axis('off')
#plt.show()
max_score, righteye_r, righteye_c, righteye_scale, righteye_response_map = pyramid_score (image, righteye_feature, righteye_shape, stepSize = 20,scale=0.9, pixel_per_cell=2)

righteye_response_map = resize(righteye_response_map, face_response_map.shape)
#plt.axis('off')
#plt.show()
max_score, nose_r, nose_c, nose_scale, nose_response_map = pyramid_score (image, nose_feature, nose_shape, stepSize = 20,scale=0.9, pixel_per_cell = 2)

nose_response_map = resize(nose_response_map, face_response_map.shape)

nose_response_map = resize(nose_response_map, face_response_map.shape)
#plt.imshow(nose_response_map,cmap='viridis', interpolation='nearest')
#plt.axis('off')
#plt.show()
max_score, mouth_r, mouth_c, mouth_scale, mouth_response_map =pyramid_score (image, mouth_feature, mouth_shape, stepSize = 20,scale=0.9, pixel_per_cell = 2)

mouth_response_map = resize(mouth_response_map, face_response_map.shape)
#plt.imshow(mouth_response_map,cmap='viridis', interpolation='nearest')
#plt.axis('off')
#plt.show()
print('2:',face_response_map.shape)
face_heatmap_shifted=shift_heatmap(face_response_map,[0,0])
lefteye_heatmap_shifted=shift_heatmap(lefteye_response_map,lefteye_mu)
righteye_heatmap_shifted=shift_heatmap(righteye_response_map,righteye_mu)
nose_heatmap_shifted=shift_heatmap(nose_response_map,nose_mu)
mouth_heatmap_shifted=shift_heatmap(mouth_response_map,mouth_mu)
'''
plt.subplot(2,2,1)
plt.imshow(lefteye_heatmap_shifted,cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(righteye_heatmap_shifted,cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(nose_heatmap_shifted,cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(mouth_heatmap_shifted,cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.show()
'''
heatmap_face= face_heatmap_shifted
heatmaps = [lefteye_heatmap_shifted,
           righteye_heatmap_shifted,
           nose_heatmap_shifted,
           mouth_heatmap_shifted]
sigmas=[lefteye_std,righteye_std,nose_std,mouth_std]
print('1:',heatmap_face.shape)
heatmap,i,j=gaussian_heatmap(heatmap_face,heatmaps,sigmas)
fig,ax = plt.subplots(1)
rect = patches.Rectangle((j-winW//2, i-winH//2),winW,winH,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)

plt.imshow(heatmap,cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.show()

fig,ax = plt.subplots(1)
rect = patches.Rectangle((j-winW//2, i-winH//2),winW,winH,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)

plt.imshow(resize(image,heatmap.shape),cmap='gray')
plt.axis('off')
plt.title('Result')
plt.show()

image_path = 'image_0002.jpg'
image = io.imread(image_path, as_grey=True)
plt.imshow(image,cmap='gray')
plt.show()
mage_path = 'image_0002.jpg'
image = io.imread(image_path, as_grey=True)
heatmap = get_heatmap(image, face_feature, face_shape, detectors_list, parts)

plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
plt.show()