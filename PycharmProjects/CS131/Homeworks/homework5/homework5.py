import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color
from time import time
from matplotlib import rc
from Homeworks.homework5.segmentation import *
from Homeworks.homework5.utils import *

#Generate random data points for clustering
#Cluster 1
'''
mean1=[-1,0]
conv1=[[0.1,0],[0,0.1]]
X1=np.random.multivariate_normal(mean1,conv1,100)

# Cluster 2
mean2 = [0, 1]
cov2 = [[0.1, 0], [0, 0.1]]
X2 = np.random.multivariate_normal(mean2, cov2, 100)

# Cluster 3
mean3 = [1, 0]
cov3 = [[0.1, 0], [0, 0.1]]
X3 = np.random.multivariate_normal(mean3, cov3, 100)

# Cluster 4
mean4 = [0, -1]
cov4 = [[0.1, 0], [0, 0.1]]
X4 = np.random.multivariate_normal(mean4, cov4, 100)


# Merge two sets of data points
X = np.concatenate((X1, X2, X3, X4))
'''
########################################################################################################################(1 Clustering Algorithms)
# Plot data points
#plt.scatter(X[:, 0], X[:, 1])
#plt.axis('equal')
#plt.show()
''''
np.random.seed(0)
start = time()
assignments = kmeans(X, 4)
end = time()

kmeans_runtime = end - start

print("kmeans running time: %f seconds." % kmeans_runtime)

for i in range(4):
    cluster_i = X[assignments==i]
    plt.scatter(cluster_i[:, 0], cluster_i[:, 1])

plt.axis('equal')
plt.show()
'''
#np.random.seed(0)
#start = time()
#assignments = kmeans_fast(X, 4)
#end = time()
#下面使用的是快速版的kmeans
#kmeans_fast_runtime = end - start
#print("kmeans running time: %f seconds." % kmeans_fast_runtime)
#print("%f times faster!" % (kmeans_runtime / kmeans_fast_runtime))

#for i in range(4):
#    cluster_i = X[assignments==i]
#    plt.scatter(cluster_i[:, 0], cluster_i[:, 1])

#plt.axis('equal')
#plt.show()
########################################################################################################################(1.2 Hierarchical Agglomerative Clustering)
'''
start = time()
assignments = hierarchical_clustering(X, 4)
end = time()
print("hierarchical_clustering running time: %f seconds." % (end - start))

for i in range(4):
    cluster_i = X[assignments==i]
    plt.scatter(cluster_i[:, 0], cluster_i[:, 1])

plt.axis('equal')
plt.show()
'''
########################################################################################################################(2 Pixel-Level Features)
img=io.imread('train.jpg')
H,W,C=img.shape

#plt.imshow(img)
#plt.axis('off')
#plt.show()
########################################################################################################################(2.1 Color Features)
'''
np.random.seed(0)

features = color_features(img)

# Sanity checks
assert features.shape == (H * W, C),"Incorrect shape! Check your implementation."

assert features.dtype == np.float,"dtype of color_features should be float."
assignments = kmeans_fast(features, 8)
segments = assignments.reshape((H, W))

# Display segmentation
plt.imshow(segments, cmap='viridis')
plt.axis('off')
plt.show()
visualize_mean_color_image(img, segments)
'''
########################################################################################################################(2.2 Color and Position Features)
'''
features = color_position_features(img)

# Sanity checks
assert features.shape == (H * W, C + 2),\
    "Incorrect shape! Check your implementation."

assert features.dtype == np.float,\
    "dtype of color_features should be float."

assignments = kmeans_fast(features, 8)
segments = assignments.reshape((H, W))

# Display segmentation
plt.imshow(segments, cmap='viridis')
plt.axis('off')
plt.show()
visualize_mean_color_image(img,segments)
'''
########################################################################################################################(3 Quantitative Evaluation)
'''
mask_gt = np.zeros((100, 100))
mask = np.zeros((100, 100))

# Test compute_accracy function
mask_gt[20:50, 30:60] = 1
mask[30:50, 30:60] = 1

accuracy = compute_accuracy(mask_gt, mask)

print('Accuracy: %0.2f' % (accuracy))
if accuracy != 0.97:
    print('Check your implementation!')

plt.subplot(121)
plt.imshow(mask_gt,cmap='gray')
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(122)
plt.imshow(mask,cmap='gray')
plt.title('Estimate')
plt.axis('off')

plt.show()
'''
# Load a small segmentation dataset
imgs, gt_masks = load_dataset('./data')

# Set the parameters for segmentation.
num_segments = 3
clustering_fn = kmeans_fast
feature_fn = color_features
scale = 0.5

mean_accuracy = 0.0

segmentations = []

for i, (img, gt_mask) in enumerate(zip(imgs, gt_masks)):
    # Compute a segmentation for this image
    segments = compute_segmentation(img, num_segments,
                                    clustering_fn=clustering_fn,
                                    feature_fn=feature_fn,
                                    scale=scale)

    segmentations.append(segments)

    # Evaluate segmentation
    accuracy = evaluate_segmentation(gt_mask, segments)

    print('Accuracy for image %d: %0.4f' % (i, accuracy))
    mean_accuracy += accuracy

mean_accuracy = mean_accuracy / len(imgs)
print('Mean accuracy: %0.4f' % mean_accuracy)
N = len(imgs)
plt.figure(figsize=(15,60))
for i in range(N):

    plt.subplot(N, 3, (i * 3) + 1)
    plt.imshow(imgs[i])
    plt.axis('off')

    plt.subplot(N, 3, (i * 3) + 2)
    plt.imshow(gt_masks[i])
    plt.axis('off')

    plt.subplot(N, 3, (i * 3) + 3)
    plt.imshow(segmentations[i], cmap='viridis')
    plt.axis('off')

plt.show()
########################################################################################################################(1.2 Hierarchical Agglomerative Clustering)
########################################################################################################################(3 Quantitative Evaluation)
########################################################################################################################(3 Quantitative Evaluation)
########################################################################################################################(3 Quantitative Evaluation)


