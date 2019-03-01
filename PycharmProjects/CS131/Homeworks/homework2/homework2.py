import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from time import time
from Homeworks.homework2.edge import *
'''Part 1: Canny Edge Detector'''
##############################################################################################(1.1 Smoothing)
# Define 3x3 Gaussian kernel with std = 1
'''test the function of gaussian_kernel'''
#kernel = gaussian_kernel(3, 1)
#kernel_test = np.array(
#    [[ 0.05854983, 0.09653235, 0.05854983],
#     [ 0.09653235, 0.15915494, 0.09653235],
#     [ 0.05854983, 0.09653235, 0.05854983]]
#)

# Test Gaussian kernel
#if not np.allclose(kernel, kernel_test):
#    print('Incorrect values! Please check your implementation.')
'''test the function of conv'''
# Test with different kernel_size and sigma
kernel_size = 5
sigma = 1.4

# Load image
#img = io.imread('iguana.png', as_gray=True)

# Define 5x5 Gaussian kernel with std = sigma
#kernel = gaussian_kernel(kernel_size, sigma)

# Convolve image with kernel to achieve smoothed effect
#smoothed = conv(img, kernel)

#plt.subplot(1,2,1)
#plt.imshow(img,cmap='gray')
#plt.title('Original image')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(smoothed,cmap='gray')
#plt.title('Smoothed image')
#plt.axis('off')

#plt.show()
##############################################################################################(1.2 Finding gradients)
# Test input
#I = np.array(
#    [[0, 0, 0],
#     [0, 1, 0],
#     [0, 0, 0]]
#)

# Expected outputs
#I_x_test = np.array(
#    [[0, 0, 0],
#     [0.5, 0, -0.5],
#     [0, 0, 0]]
#)

#I_y_test = np.array(
#    [[0, 0.5, 0],
#     [0, 0, 0],
#     [0, -0.5, 0]]
#)

# Compute partial derivatives
#I_x = partial_x(I)
#I_y = partial_y(I)

# Test correctness of partial_x and partial_y
#if not np.all(I_x == I_x_test):
#    print('partial_x incorrect')

#if not np.all(I_y == I_y_test):
#    print('partial_y incorrect')
#Gx = partial_x(smoothed)
#Gy = partial_y(smoothed)

#plt.subplot(1,2,1)
#plt.imshow(Gx,cmap='gray')
#plt.title('Derivative in x direction')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(Gy,cmap='gray')
#plt.title('Derivative in y direction')
#plt.axis('off')

#plt.show()
#G, theta = gradient(smoothed)

#if not np.all(G >= 0):
#    print('Magnitude of gradients should be non-negative.')

#if not np.all((theta >= 0) * (theta < 360)):
#    print('Direction of gradients should be in range 0 <= theta < 360')

#plt.imshow(G,cmap='gray')
#plt.title('Gradient magnitude')
#plt.axis('off')
#plt.show()
# Test input
#g = np.array(
#    [[0.4, 0.5, 0.6],
#     [0.3, 0.5, 0.7],
#     [0.4, 0.5, 0.6]]
#)

# Print out non-maximum suppressed output
# varying theta
#for angle in range(0, 180, 45):
#    print('Thetas:', angle)
#    t = np.ones((3, 3)) * angle # Initialize theta
#    print(non_maximum_suppression(g, t))
#nms = non_maximum_suppression(G, theta)
#plt.imshow(G,cmap='gray')
#plt.figure()
#plt.imshow(nms,cmap='gray')
#plt.figure()
#plt.title('Non-maximum suppressed')
#plt.axis('off')
#plt.show()
#low_threshold = 0.02
#high_threshold = 0.03

#strong_edges, weak_edges = double_thresholding(nms, high_threshold, low_threshold)
#print(strong_edges)
#print(weak_edges)
#assert(np.sum(strong_edges * weak_edges) == 0)

#edges=strong_edges * 1.0 + weak_edges * 0.5

#plt.subplot(1,2,1)
#plt.imshow(strong_edges,cmap='gray')
#plt.title('Strong Edges')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(edges,cmap='gray')
#plt.title('Strong+Weak Edges')
#plt.axis('off')

#plt.show()


#edges = link_edges(strong_edges, weak_edges)

#plt.imshow(edges,cmap='gray')
#plt.axis('off')
#plt.show()
#img = io.imread('iguana.png', as_gray=True)

# Run Canny edge detector
#edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
#print(edges.shape)
#io.imsave('final.png',edges)
#plt.subplot(3,2,5)
#plt.imshow(edges,cmap='gray')
#img_orignal=io.imread('iguana_edges.png', as_gray=True)
#plt.subplot(3,2,6)
#plt.imshow(img_orignal,cmap='gray')
#plt.axis('off')
#plt.show()


'''Part2: Lane Detection'''
##############################################################################################(2.1 Edge detection)
# Load image
img = io.imread('test.png', as_gray=True)

# Run Canny edge detector
edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
io.imsave('test_edges.png',edges)
#io.imsave('road_edges.jpg',edges)

#plt.subplot(211)
#plt.imshow(img,cmap='gray')
#plt.axis('off')
#plt.title('Input Image')

#plt.subplot(212)
#plt.imshow(edges,cmap='gray')
#plt.axis('off')
#plt.title('Edges')
#plt.show()
##############################################################################################(2.2 Extracting region of interest)
H, W = img.shape

# Generate mask for ROI (Region of Interest)
mask = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        if i > (H / W) * j and i > -(H / W) * j + H:
            mask[i, j] = 1

# Extract edges in ROI
roi = edges * mask

#plt.subplot(1,2,1)
#plt.imshow(mask)
#plt.title('Mask')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(roi)
#plt.title('Edges in ROI')
#plt.axis('off')
#plt.show()
acc, rhos, thetas = hough_transform(roi)

# Coordinates for right lane
xs_right = []
ys_right = []

# Coordinates for left lane
xs_left = []
ys_left = []

for i in range(20):
    idx = np.argmax(acc)
    r_idx = idx // acc.shape[1]
    t_idx = idx % acc.shape[1]
    acc[r_idx, t_idx] = 0  # Zero out the max value in accumulator

    rho = rhos[r_idx]
    theta = thetas[t_idx]

    # Transform a point in Hough space to a line in xy-space.
    a = - (np.cos(theta) / np.sin(theta))  # slope of the line
    b = (rho / np.sin(theta))  # y-intersect of the line

    # Break if both right and left lanes are detected
    if xs_right and xs_left:
        break

    if a < 0:  # Left lane
        if xs_left:
            continue
        xs = xs_left
        ys = ys_left
    else:  # Right Lane
        if xs_right:
            continue
        xs = xs_right
        ys = ys_right

    for x in range(img.shape[1]):
        y = a * x + b
        if y > img.shape[0] * 0.6 and y < img.shape[0]:
            xs.append(x)
            ys.append(int(round(y)))

plt.imshow(img,cmap='gray')
plt.plot(xs_left, ys_left, linewidth=5.0)
plt.plot(xs_right, ys_right, linewidth=5.0)
plt.axis('off')
plt.show()