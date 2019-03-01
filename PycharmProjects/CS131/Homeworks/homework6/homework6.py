from time import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io
from Homeworks.homework6.compression import *
from Homeworks.homework6.features import *
from Homeworks.homework6.k_nearest_neighbor import *
from Homeworks.homework6.utils import *

########################################################################################################################(Part 1 - Image Compression)
image=io.imread('pitbull.jpg',as_gray=True)
#plt.imshow(image,cmap='gray')
#plt.axis('off')
#plt.show()

#compressed_image, compressed_size = compress_image(image, 100)
#compression_ratio = compressed_size / image.size
#print('Original image shape:', image.shape)
#print('Compressed size: %d' % compressed_size)
#print('Compression ratio: %.3f' % compression_ratio)
#assert compressed_size == 298500

'''
n_values=[10,50,100]
for n in n_values:
    # Compress the image using `n` singular values
    compressed_image, compressed_size = compress_image(image, n)

    compression_ratio = compressed_size / image.size

    print("Data size (original): %d" % (image.size))
    print("Data size (compressed): %d" % compressed_size)
    print("Compression ratio: %f" % (compression_ratio))

    plt.imshow(compressed_image, cmap='gray')
    title = "n = %s" % n
    plt.title(title)
    plt.axis('off')
    plt.show()
'''

X_train, y_train, classes_train = load_dataset('faces', train=True, as_grey=True)
X_test, y_test, classes_test = load_dataset('faces', train=False, as_grey=True)

assert classes_train == classes_test
classes = classes_train
num_classes = len(classes)
'''

print('Class names:', classes)
print('Training data shape:', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape:', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
num_classes = len(classes)
samples_per_class = 10
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)      #函数np.flatnonzero返回的是数组中非零值的位置，并且是每行每行的扫描，不断累加
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx],cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(y)
plt.show()
'''

# Flatten the image data into rows
# we now have one 4096 dimensional featue vector for each example
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

dists = compute_distances(X_test, X_train)
'''
assert dists.shape == (160, 800)
print("dists shape:", dists.shape)

# We use k = 1 (which corresponds to only taking the nearest neighbor to decide)
y_test_pred = predict_labels(dists, y_train, k=1)

# Compute and print the fraction of correctly predicted examples
num_test = y_test.shape[0]
num_correct = int(np.sum(y_test_pred == y_test))
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
'''

########################################################################################################################(Cross-Validation)
# Step 2: split the data into 5 folds to perform cross-validation.
'''
num_folds = 5

X_trains, y_trains, X_vals, y_vals = split_folds(X_train, y_train, num_folds)

assert X_trains.shape == (5, 640, 4096)
assert y_trains.shape == (5, 640)
assert X_vals.shape == (5, 160, 4096)
assert y_vals.shape == (5, 160)
# Step 3: Measure the mean accuracy for each value of `k`

# List of k to choose from
k_choices = list(range(5, 101, 5))

# Dictionnary mapping k values to accuracies
# For each k value, we will have `num_folds` accuracies to compute
# k_to_accuracies[1] will be for instance [0.22, 0.23, 0.19, 0.25, 0.20] for 5 folds
k_to_accuracies = {}

for k in k_choices:
    print("Running for k=%d" % k)
    accuracies = []
    for i in range(num_folds):
        # Make predictions
        fold_dists = compute_distances(X_vals[i], X_trains[i])
        y_pred = predict_labels(fold_dists, y_trains[i], k)

        # Compute and print the fraction of correctly predicted examples
        num_correct = np.sum(y_pred == y_vals[i])
        accuracy = float(num_correct) / len(y_vals[i])
        accuracies.append(accuracy)

    k_to_accuracies[k] = accuracies
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 26% accuracy on the test data.


# YOUR CODE HERE
# Choose the best k based on the cross validation above
best_k=int(np.argmax(accuracies_mean)+1)*5
# END YOUR CODE

y_test_pred = predict_labels(dists, y_train, k=best_k)

# Compute and display the accuracy
num_test=X_test.shape[0]
num_correct = int(np.sum(y_test_pred == y_test))
accuracy = float(num_correct) / num_test
print('For k = %d, got %d / %d correct => accuracy: %f' % (best_k, num_correct, num_test, accuracy))
'''
########################################################################################################################(Part 3: PCA)
pca=PCA()
#e_vecs, e_vals = pca._eigen_decomp(X_train - X_train.mean(axis=0))

#print(e_vals.shape)
#print(e_vecs.shape)
# Perform SVD on directly on the training data.

u, s = pca._svd(X_train - X_train.mean(axis=0))

# Dimensionality reduction by projecting the data onto
# lower dimensional subspace spanned by k principal components

# To visualize, we will project in 2 dimensions
n_components = 2
pca.fit(X_train)
X_proj = pca.transform(X_train, n_components)
''''
# Plot the top two principal components
for y in np.unique(y_train):
    plt.scatter(X_proj[y_train == y, 0], X_proj[y_train == y, 1], label=classes[y])

plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.legend()
plt.show()
'''
'''
plt.rcParams['figure.dpi'] = 750 #分辨率
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(pca.W_pca[:, i].reshape(64, 64),cmap='gray')
    #plt.title("%.2f" % s[i])
plt.show()
# Reconstruct data with principal components
n_components = 100  # Experiment with different number of components.
X_proj = pca.transform(X_train, n_components)
X_rec = pca.reconstruct(X_proj)

print(X_rec.shape)
print(classes)

# Visualize reconstructed faces
samples_per_class = 10
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow((X_rec[idx]).reshape((64, 64)),cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(y)
plt.show()


# Plot reconstruction errors for different k
N=X_train.shape[0]
d=X_train.shape[1]

ns=range(1,d,100)
errors=[]
for n in ns:
    X_proj = pca.transform(X_train, n)
    X_rec = pca.reconstruct(X_proj)

    # Compute reconstruction error
    error = np.mean((X_rec - X_train) ** 2)
    errors.append(error)

plt.plot(ns,errors)
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.show()
'''

########################################################################################################################(Part 4 - Fisherface: Linear Discriminant Analysis)
lda=LDA()
N=X_train.shape[0]
c=num_classes

pac=PCA()
pca.fit(X_train)
X_train_pca=pca.transform(X_train,N-c)
X_test_pca=pca.transform(X_test,N-c)
'''
# Compute within-class scatter matrix
S_W = lda._within_class_scatter(X_train_pca, y_train)
print(S_W.shape)

# Compute between-class scatter matrix
S_B = lda._between_class_scatter(X_train_pca, y_train)
print(S_B.shape)
'''


'''
lda.fit(X_train_pca, y_train)
# Dimensionality reduction by projecting the data onto
# lower dimensional subspace spanned by k principal components
n_components = 2
X_proj = lda.transform(X_train_pca, n_components)
X_test_proj = lda.transform(X_test_pca, n_components)

# Plot the top two principal components on the training set
for y in np.unique(y_train):
    plt.scatter(X_proj[y_train == y, 0], X_proj[y_train == y, 1], label=classes[y])

plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.legend()
plt.title("Training set")
plt.show()

# Plot the top two principal components on the test set
for y in np.unique(y_test):
    plt.scatter(X_test_proj[y_test == y, 0], X_test_proj[y_test == y, 1], label=classes[y])

plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.legend()
plt.title("Test set")
plt.show()
'''
num_folds = 5

X_trains, y_trains, X_vals, y_vals = split_folds(X_train, y_train, num_folds)

k_choices = [1, 5, 10, 20]
n_choices = [5, 10, 20, 50, 100, 200, 500]
pass

# n_k_to_accuracies[(n, k)] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of n and k.
n_k_to_accuracies = defaultdict(list)

for i in range(num_folds):
    # Fit PCA
    pca = PCA()
    pca.fit(X_trains[i])

    N = len(X_trains[i])
    X_train_pca = pca.transform(X_trains[i], N - c)
    X_val_pca = pca.transform(X_vals[i], N - c)

    # Fit LDA
    lda = LDA()
    lda.fit(X_train_pca, y_trains[i])

    for n in n_choices:
        X_train_proj = lda.transform(X_train_pca, n)
        X_val_proj = lda.transform(X_val_pca, n)

        dists = compute_distances(X_val_proj, X_train_proj)

        for k in k_choices:
            y_pred = predict_labels(dists, y_trains[i], k=k)

            # Compute and print the fraction of correctly predicted examples
            num_correct = np.sum(y_pred == y_vals[i])
            accuracy = float(num_correct) / len(y_vals[i])
            n_k_to_accuracies[(n, k)].append(accuracy)

for n in n_choices:
    print()
    for k in k_choices:
        accuracies = n_k_to_accuracies[(n, k)]
        print("For n=%d, k=%d: average accuracy is %f" % (n, k, np.mean(accuracies)))