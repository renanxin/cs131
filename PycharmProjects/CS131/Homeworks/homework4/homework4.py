import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import color
from time import time
from skimage import io,util
from Homeworks.homework4.seam_carving import *

########################################################################################################################(Image Reducing using Seam Carving)
#Load image
img=io.imread('broadway_tower.jpg')
img=util.img_as_float(img)

#plt.title('Original Image')
#plt.imshow(img)
#plt.show()
#test_img = np.array([[1.0, 2.0, 1.5],
#                     [3.0, 1.0, 2.0],
#                     [4.0, 0.5, 3.0]])
#test_img = np.stack([test_img] * 3, axis=2)
#assert test_img.shape == (3, 3, 3)

# Compute energy function
'''
test_energy = energy_function(test_img)

solution_energy = np.array([[3.0, 1.25,  1.0],
                            [3.5, 1.25, 1.75],
                            [4.5,  1.0,  3.5]])

print("Image (channel 0):")
print(test_img[:, :, 0])

print("Energy:")
print(test_energy)
print("Solution energy:")
print(solution_energy)

assert np.allclose(test_energy, solution_energy)
# Compute energy function
start = time()
energy = energy_function(img)
end = time()

print("Computing energy function: %f seconds." % (end - start))

plt.title('Energy')
plt.axis('off')
plt.imshow(energy,cmap='gray')
plt.show()
'''
#energy=energy_function(img)

#start = time()
#vcost, vpaths = compute_cost(img,energy=energy, axis=1)  # don't need the first argument for compute_cost
#end = time()

#print("Computing vertical cost map: %f seconds." % (end - start))

#plt.title('Vertical Cost Map')
#plt.axis('off')
#plt.imshow(vcost, cmap='inferno')
#plt.show()
#start = time()
#hcost, hpaths = compute_cost(img, energy, axis=0)
#end = time()

#print("Computing horizontal cost map: %f seconds." % (end - start))

#plt.title('Horizontal Cost Map')
#plt.axis('off')
#plt.imshow(hcost, cmap='inferno')
#plt.show()
#start = time()
#end = np.argmin(vcost[-1])
#seam_energy = vcost[-1, end]
#seam = backtrack_seam(vpaths, end)
#end = time()

#print("Backtracking optimal seam: %f seconds." % (end - start))
#print('Seam Energy:', seam_energy)

# Visualize seam
#vseam = np.copy(img)
#for row in range(vseam.shape[0]):
#    vseam[row, seam[row], :] = np.array([1.0, 0, 0])

#plt.title('Vertical Seam')
#plt.axis('off')
#plt.imshow(vseam)
#plt.show()

# Reduce image height
H, W, _ = img.shape
#H_new = 300

#start = time()
#out = reduce(img, H_new, axis=0)
#end = time()

#print("Reducing height from %d to %d: %f seconds." % (H, H_new, end - start))

#plt.subplot(1, 2, 1)
#plt.title('Original')
#plt.imshow(img)

#plt.subplot(1, 2, 2)
#plt.title('Resized')
#plt.imshow(out)

#plt.show()

########################################################################################################################(Image Enlarging)
# Let's first test with a small example
W_new = 800

# This is a naive implementation of image enlarging
# which iteratively computes energy function, finds optimal seam
# and duplicates it.
# This process will a stretching artifact by choosing the same seam
#start = time()
#enlarged = enlarge_naive(img, W_new)
#end = time()

# Can take around 20 seconds
#print("Enlarging(naive) height from %d to %d: %f seconds." % (W, W_new, end - start))

#plt.imshow(enlarged)
#plt.show()

# Alternatively, find k seams for removal and duplicate them.
#start = time()
#seams = find_seams(img, W_new - W)
#end = time()

# Can take around 10 seconds
#print("Finding %d seams: %f seconds." % (W_new - W, end - start))

#plt.imshow(seams, cmap='viridis')
#plt.show()
#W_new = 800

#start = time()
#out = enlarge(img, W_new)
#end = time()

# Can take around 20 seconds
#print("Enlarging width from %d to %d: %f seconds." % (W, W_new, end - start))

#plt.subplot(2, 1, 1)
#plt.title('Original')
#plt.imshow(img)

#plt.subplot(2, 1, 2)
#plt.title('Resized')
#plt.imshow(out)

#plt.show()
# Reduce image width
#H, W, _ = img.shape
#W_new = 400

#start = time()
#out = reduce(img, W_new)
#end = time()

#print("Normal reduce width from %d to %d: %f seconds." % (W, W_new, end - start))

#start = time()
#out_fast = reduce_fast(img, W_new)
#end = time()

#print("Faster reduce width from %d to %d: %f seconds." % (W, W_new, end - start))

#assert np.allclose(out, out_fast), "Outputs don't match."


#plt.subplot(3, 1, 1)
#plt.title('Original')
#plt.imshow(img)

#plt.subplot(3, 1, 2)
#plt.title('Resized')
#plt.imshow(out)

#plt.subplot(3, 1, 3)
#plt.title('Faster resized')
#plt.imshow(out)

#plt.show()
# Load image
#img2 = io.imread('wave.jpg')
#img2 = util.img_as_float(img2)

#plt.title('Original Image')
#plt.imshow(img2)
#plt.show()
#out = reduce(img2, 300)
#plt.imshow(out)
#plt.show()
#out = enlarge(img2, 800)
#plt.imshow(out)
#plt.show()
'''
img_yolo = io.imread('yolo.jpg')
img_yolo = util.img_as_float(img_yolo)

plt.title('Original Image')
plt.imshow(img_yolo)
plt.show()
energy = energy_function(img_yolo)

out, _ = compute_cost(img_yolo, energy)
plt.subplot(1, 2, 1)
plt.imshow(out, cmap='inferno')
plt.title("Normal cost function")

out, _ = compute_forward_cost(img_yolo, energy)
plt.subplot(1, 2, 2)
plt.imshow(out, cmap='inferno')
plt.title("Forward cost function")

plt.show()
out = reduce(img_yolo, 200, axis=0)
plt.imshow(out)
plt.show()
'''
# Load image
image = io.imread('wyeth.jpg')
image = util.img_as_float(image)

mask = io.imread('wyeth_mask.jpg', as_gray=True)
mask = util.img_as_bool(mask)

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title('Mask of the object to remove')
plt.imshow(mask)

plt.show()
out = remove_object(image, mask)

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image)

plt.subplot(2, 2, 2)
plt.title('Mask of the object to remove')
plt.imshow(mask)

plt.subplot(2, 2, 3)
plt.title('Image with object removed')
plt.imshow(out)

plt.show()