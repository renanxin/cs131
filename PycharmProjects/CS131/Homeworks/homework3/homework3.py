import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.io import imread
import matplotlib.pyplot as plt
from time import time
from Homeworks.homework3.panorama import *
from Homeworks.homework3.utils import *
from skimage import io

###################################################################################(Part 1 Harris Corner Detector)
#img = imread('shudu.png', as_gray=True)
#response=harris_corners(img)
# Display corner response
#plt.subplot(1,2,1)
#plt.imshow(response,cmap='gray')
#plt.axis('off')
#plt.title('Harris Corner Response')

#plt.subplot(1,2,2)
#plt.imshow(imread('solution_harris.png',as_gray=True),cmap='gray')
#plt.axis('off')
#plt.title('Harris Corner Solution')

#plt.show()
# Perform non-maximum suppression in response map
# and output corner coordiantes
#corners = corner_peaks(response, threshold_rel=0.01)

# Display detected corners
#plt.imshow(img,cmap='gray')
#plt.scatter(corners[:,1], corners[:,0], marker='x',)
#plt.axis('off')
#plt.title('Detected Corners')
#plt.show()


'''Part 2 Describing and Matching Keypoints'''
###################################################################################(Part 2.1 Creating Descriptors)
#img1 = imread('yosemite1.jpg', as_gray=True)
#img2 = imread('yosemite2.jpg', as_gray=True)

# Detect keypoints in two images
#keypoints1 = corner_peaks(harris_corners(img1, window_size=3),threshold_rel=0.05,exclude_border=8)
#keypoints2 = corner_peaks(harris_corners(img2, window_size=3),threshold_rel=0.05,exclude_border=8)

# Display detected keypoints
#plt.subplot(1,2,1)
#plt.imshow(img1,cmap='gray')
#plt.scatter(keypoints1[:,1], keypoints1[:,0], marker='x')
#plt.axis('off')
#plt.title('Detected Keypoints for Image 1')

#plt.subplot(1,2,2)
#plt.imshow(img2,cmap='gray')
#plt.scatter(keypoints2[:,1], keypoints2[:,0], marker='x')
#plt.axis('off')
#plt.title('Detected Keypoints for Image 2')
#plt.show()
#patch_size = 5

# Extract features from the corners
#desc1 = describe_keypoints(img1, keypoints1,desc_func=simple_descriptor,patch_size=patch_size)
#desc2 = describe_keypoints(img2, keypoints2,desc_func=simple_descriptor,patch_size=patch_size)

# Match descriptors in image1 to those in image2
#matches = match_descriptors(desc1, desc2, 0.7)
# Plot matches
#fig, ax = plt.subplots(1, 1, figsize=(15, 12))
#ax.axis('off')
#plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
#plt.show()
#plt.imshow(imread('solution_simple_descriptor.png'),cmap='gray')
#plt.axis('off')
#plt.title('Matched Simple Descriptor Solution')
#plt.show()
# Extract matched keypoints
#p1 = keypoints1[matches[:,0]]
#p2 = keypoints2[matches[:,1]]

# Find affine transformation matrix H that maps p2 to p1
#H = fit_affine_matrix(p1, p2)   #得到最好的变化矩阵

#output_shape, offset = get_output_space(img1, [img2], [H])
#print("Output shape:", output_shape)
#print("Offset:", offset)


# Warp images into output sapce
#img1_warped = warp_image(img1, np.eye(3), output_shape, offset)
#img1_mask = (img1_warped != -1) # Mask == 1 inside the image
#img1_warped[~img1_mask] = 0     # Return background values to 0

#img2_warped = warp_image(img2, H, output_shape, offset)
#img2_mask = (img2_warped != -1) # Mask == 1 inside the image
#img2_warped[~img2_mask] = 0     # Return background values to 0

# Plot warped images
#plt.subplot(1,2,1)
#plt.imshow(img1_warped,cmap='gray')
#plt.title('Image 1 warped')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(img2_warped,cmap='gray')
#plt.title('Image 2 warped')
#plt.axis('off')

#plt.show()
#merged = img1_warped + img2_warped

# Track the overlap by adding the masks together
#overlap = (img1_mask * 1.0 +  # Multiply by 1.0 for bool -> float conversionimg2_mask)

# Normalize through division by `overlap` - but ensure the minimum is 1
#normalized = merged / np.maximum(overlap, 1)
#plt.imshow(normalized,cmap='gray')
#plt.axis('off')
#plt.show()
'''Part 4 RANSAC'''
#H, robust_matches = ransac(keypoints1, keypoints2, matches, threshold=1)

# Visualize robust matches
#fig, ax = plt.subplots(1, 1, figsize=(15, 12))
#plot_matches(ax, img1, img2, keypoints1, keypoints2, robust_matches)
#plt.axis('off')
#plt.show()

#plt.imshow(imread('solution_ransac.png'))
#plt.axis('off')
#plt.title('RANSAC Solution')
#plt.show()
#output_shape,offset=get_output_space(img1,[img2],[H])

#img1_warped=warp_image(img1,np.eye(3),output_shape,offset)
#img1_mask=(img1_warped!=-1)
#img1_warped[~img1_mask]=0

#img2_warped=warp_image(img2,H,output_shape,offset)
#img2_mask=(img2_warped!=-1)
#img2_warped[~img2_mask]=0

#plt.subplot(1,2,1)
#plt.imshow(img1_warped,cmap='gray')
#plt.title('Image 1 warped')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(img2_warped,cmap='gray')
#plt.title('Image 2 warped')
#plt.axis('off')

#plt.show()
#merged=img1_warped+img2_warped
#overlap=(img1_mask*1.0+img2_mask)
#normalized=merged/np.maximum(overlap,1)
#plt.imshow(normalized,cmap='gray')
#plt.axis('off')
#plt.show()

#plt.imshow(imread('solution_ransac_panorama.png'))
#plt.axis('off')
#plt.title('RANSAC Panorama Solution')
#plt.show()

'''Part 5 Histogram of Oriented Gradients (HOG)'''
#img1 = imread('uttower1.jpg', as_gray=True)
#img2 = imread('uttower2.jpg', as_gray=True)

# Detect keypoints in both images
#keypoints1 = corner_peaks(harris_corners(img1, window_size=3),
#                          threshold_rel=0.05,
#                          exclude_border=8)
#keypoints2 = corner_peaks(harris_corners(img2, window_size=3),
#                          threshold_rel=0.05,
#                          exclude_border=8)
# Extract features from the corners

#desc1 = describe_keypoints(img1, keypoints1,
#                           desc_func=hog_descriptor,
#                           patch_size=16)
#desc2 = describe_keypoints(img2, keypoints2,
#                           desc_func=hog_descriptor,
#                           patch_size=16)
# Match descriptors in image1 to those in image2
#matches = match_descriptors(desc1, desc2, 0.7)

# Plot matches
#fig, ax = plt.subplots(1, 1, figsize=(15, 12))
#ax.axis('off')
#plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
#plt.show()
#plt.imshow(imread('solution_hog.png'),cmap='gray')
#plt.axis('off')
#plt.title('HOG descriptor Solution')
#plt.show()
#H, robust_matches = ransac(keypoints1, keypoints2, matches, threshold=1)
#output_shape, offset = get_output_space(img1, [img2], [H])
#print(output_shape, offset)

# Warp images into output sapce
#img1_warped = warp_image(img1, np.eye(3), output_shape, offset)
#img1_mask = (img1_warped != -1) # Mask == 1 inside the image
#img1_warped[~img1_mask] = 0     # Return background values to 0

#img2_warped = warp_image(img2, H, output_shape, offset)
#img2_mask = (img2_warped != -1) # Mask == 1 inside the image
#img2_warped[~img2_mask] = 0     # Return background values to 0

# Plot warped images
#plt.subplot(1,2,1)
#plt.imshow(img1_warped,cmap='gray')
#plt.title('Image 1 warped')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(img2_warped,cmap='gray')
#plt.title('Image 2 warped')
#plt.axis('off')

#plt.show()
#merged = img1_warped + img2_warped

#overlap = (img1_mask * 1.0 + img2_mask)
#Gx = filters.sobel_v(overlap)
#Gy = filters.sobel_h(overlap)
#G = np.sqrt(Gx ** 2 + Gy ** 2)
#output = merged / np.maximum(overlap, 1)
#idx = np.where(G > 0)
#for x, y in zip(idx[0], idx[1]):
#    if y < 458:
#        output[x, y] = np.mean(output[x, 0: y]) # TODO
### END YOUR CODE

img1 = imread('yosemite1.jpg', as_gray=True)
img2 = imread('yosemite2.jpg', as_gray=True)
img3 = imread('yosemite3.jpg', as_gray=True)
img4 = imread('yosemite4.jpg', as_gray=True)

# Detect keypoints in each image
keypoints1 = corner_peaks(harris_corners(img1, window_size=3),
                          threshold_rel=0.05,
                          exclude_border=8)
keypoints2 = corner_peaks(harris_corners(img2, window_size=3),
                          threshold_rel=0.05,
                          exclude_border=8)
keypoints3 = corner_peaks(harris_corners(img3, window_size=3),
                          threshold_rel=0.05,
                          exclude_border=8)
keypoints4 = corner_peaks(harris_corners(img4, window_size=3),
                          threshold_rel=0.05,
                          exclude_border=8)

patch_size=16
# Describe keypoints
desc1 = describe_keypoints(img1, keypoints1,
                           desc_func=simple_descriptor,
                           patch_size=patch_size)
desc2 = describe_keypoints(img2, keypoints2,
                           desc_func=simple_descriptor,
                           patch_size=patch_size)
desc3 = describe_keypoints(img3, keypoints3,
                           desc_func=simple_descriptor,
                           patch_size=patch_size)
desc4 = describe_keypoints(img4, keypoints4,
                           desc_func=simple_descriptor,
                           patch_size=patch_size)

# Match keypoints in neighboring images
matches12 = match_descriptors(desc1, desc2, 0.7)
matches23 = match_descriptors(desc2, desc3, 0.7)
matches34 = match_descriptors(desc3, desc4, 0.7)

### YOUR CODE HERE
H12, _ = ransac(keypoints1, keypoints2, matches12)
H23, _ = ransac(keypoints2, keypoints3, matches23)
H34, _ = ransac(keypoints3, keypoints4, matches34)
# Take image2 as reference image
output_shape, offset = get_output_space(img2, [img1, img3, img4], [np.linalg.inv(H12), H23, H23.dot(H34)])

img1_warped = warp_image(img1, np.linalg.inv(H12), output_shape, offset)
img1_mask = (img1_warped != -1)
img1_warped[~img1_mask] = 0

img2_warped = warp_image(img2, np.eye(3), output_shape, offset)
img2_mask = (img2_warped != -1)
img2_warped[~img2_mask] = 0

img3_warped = warp_image(img3, H23, output_shape, offset)
img3_mask = (img3_warped != -1)
img3_warped[~img3_mask] = 0

img4_warped = warp_image(img4, H23.dot(H34), output_shape, offset)
img4_mask = (img4_warped != -1)
img4_warped[~img4_mask] = 0
### END YOUR CODE
# Plot warped images
merged = img1_warped + img2_warped + img3_warped + img4_warped

# Track the overlap by adding the masks together
overlap = (img2_mask * 1.0 +  # Multiply by 1.0 for bool -> float conversion
           img1_mask + img3_mask + img4_mask)
# Normalize through division by `overlap` - but ensure the minimum is 1
normalized = merged / np.maximum(overlap, 1)
plt.imshow(normalized,cmap='gray')
plt.axis('off')
plt.show()
normalized/=np.maximum(np.max(normalized),-np.min(normalized))
io.imsave('complete_picture.jpg',normalized)
