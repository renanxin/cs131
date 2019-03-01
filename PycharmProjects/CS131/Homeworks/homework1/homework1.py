import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from time import time
from Homeworks.homework1.filters import *
from scipy.misc import imread
#############################################################################################(1.1)
'''证明略，主要是通过换元法'''
#############################################################################################(1.2)
'''证明略'''
#############################################################################################(1.3)
#img=io.imread('dog.jpg',as_gray=True)
#img=zero_pad(img,10,10)
#plt.imshow(img)
#plt.axis('off')
#plt.title("Isn't he cute?")
#plt.show()
# Simple convolution kernel.
'''Test
kernel = np.array(
[
    [1,0,1],
    [0,0,0],
    [1,0,1]
])

# Create a test image: a white square in the middle
test_img = np.zeros((9, 9))
test_img[3:6, 3:6] = 1

# Run your conv_nested function on the test image
test_output = conv_nested(test_img, kernel)

# Build the expected output
expected_output = np.zeros((9, 9))
expected_output[2:7, 2:7] = 1
expected_output[4, 2:7] = 2
expected_output[2:7, 4] = 2
expected_output[4, 4] = 4

# Plot the test image
plt.subplot(1,3,1)
plt.imshow(test_img)
plt.title('Test image')
plt.axis('off')

# Plot your convolved image
plt.subplot(1,3,2)
plt.imshow(test_output)
plt.title('Convolution')
plt.axis('off')

# Plot the exepected output
plt.subplot(1,3,3)
plt.imshow(expected_output)
plt.title('Exepected output')
plt.axis('off')
plt.show()

# Test if the output matches expected output
assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."
'''
kernel=np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])
#out=conv_nested(img,kernal)
#plot original image
#plt.subplot(2,2,1)
#plt.imshow(img,cmap='gray')
#plt.title('Original')
#plt.axis('off')

# Plot your convolved image
#plt.subplot(2,2,3)
#plt.imshow(out,cmap='gray')
#plt.title('Convolution')
#plt.axis('off')

# Plot what you should get
#solution_img = io.imread('convoluted_dog.jpg', as_gray=True)
#plt.subplot(2,2,4)
#plt.imshow(solution_img,cmap='gray')
#plt.title('What you should get')
#plt.axis('off')
#plt.show()

#t0 = time()
#out_fast = conv_fast(img, kernel)
#t1 = time()
#out_nested = conv_nested(img, kernel)
#t2 = time()

# Compare the running time of the two implementations
#print("conv_nested: took %f seconds." % (t2 - t1))
#print("conv_fast: took %f seconds." % (t1 - t0))

# Plot conv_nested output
#plt.subplot(1,2,1)
#plt.imshow(out_nested,cmap='gray')
#plt.title('conv_nested')
#plt.axis('off')

# Plot conv_fast output
#plt.subplot(1,2,2)
#plt.imshow(out_fast,cmap='gray')
#plt.title('conv_fast')
#plt.axis('off')
#plt.show()

# Make sure that the two outputs are the same
#if not (np.max(out_fast - out_nested) < 1e-10):
#    print("Different outputs! Check your implementation.")
#############################################################################################(extra credit)
#############################################################################################(2.1)
# Load template and image in grayscale
'''
img = io.imread('shelf.jpg')
img_grey = io.imread('shelf.jpg', as_gray=True)
temp = io.imread('template.jpg')
temp_grey = io.imread('template.jpg', as_gray=True)

# Perform cross-correlation between the image and the template
out = zero_mean_cross_correlation(img_grey, temp_grey)

# Find the location with maximum similarity
y,x = (np.unravel_index(out.argmax(), out.shape))

# Display product template
plt.figure(figsize=(25,20))
plt.subplot(3, 1, 1)
plt.imshow(temp)
plt.title('Template')
plt.axis('off')

# Display cross-correlation output
plt.subplot(3, 1, 2)
plt.imshow(out)
plt.title('Cross-correlation (white means more correlated)')
plt.axis('off')

# Display image
plt.subplot(3, 1, 3)
plt.imshow(img)
plt.title('Result (blue marker on the detected location)')
plt.axis('off')

# Draw marker at detected location
plt.plot(x, y, 'bx', ms=40, mew=10)
plt.show()


def check_product_on_shelf(shelf, product):
    out = zero_mean_cross_correlation(shelf, product)

    # Scale output by the size of the template
    out = out / float(product.shape[0] * product.shape[1])

    # Threshold output (this is arbitrary, you would need to tune the threshold for a real application)
    out = out > 0.025

    if np.sum(out) > 0:
        print('The product is on the shelf')
    else:
        print('The product is not on the shelf')


# Load image of the shelf without the product
img2 = io.imread('shelf_soldout.jpg')
img2_grey = io.imread('shelf_soldout.jpg', as_gray=True)

plt.imshow(img)
plt.axis('off')
plt.show()
check_product_on_shelf(img_grey, temp_grey)

plt.imshow(img2)
plt.axis('off')
plt.show()
check_product_on_shelf(img2_grey, temp_grey)
'''
# Load image
img_grey = io.imread('shelf.jpg', as_gray=True)
img = io.imread('shelf_dark.jpg')
temp = io.imread('template.jpg')
temp_grey = io.imread('template.jpg', as_gray=True)

# Perform cross-correlation between the image and the template
out = normalized_cross_correlation(img_grey, temp_grey)

# Find the location with maximum similarity
y,x = (np.unravel_index(out.argmax(), out.shape))

# Display image
plt.imshow(img)
plt.title('Result (red marker on the detected location)')
plt.axis('off')

# Draw marker at detcted location
plt.plot(x, y, 'rx', ms=25, mew=5)
plt.show()