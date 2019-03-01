import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.
    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel
    Returns:
        kernel: numpy array of shape (size, size)
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k=(size-1)//2
    for i in range(size):
        for j in range(size):
            kernel[i][j]=1/(2*np.pi*sigma**2)*np.exp((-(i-k)**2-(j-k)**2)/(2*sigma**2))
    ### END YOUR CODE

    return kernel

def conv(image, kernel):
    """ An implementation of convolution filter.
    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)
    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    kernel_new=np.flip(np.flip(kernel,0),1).copy()
    for i in range(Hi):
        for j in range(Wi):
            out[i][j]=np.sum(np.multiply(kernel_new[:,:],padded[i:i+Hk,j:j+Wk]))
    ### END YOUR CODE

    return out

def partial_x(img):
    """ Computes partial x-derivative of input img.
    Hints:
        - You may use the conv function in defined in this file.
    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel=np.array([[1/2,0,-1/2]])
    out=conv(img,kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.
    Hints:
        - You may use the conv function in defined in this file.
    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel=np.array([[1/2],[0],[-1/2]])
    out=conv(img,kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W)
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx=partial_x(img)
    Gy=partial_y(img)
    G=np.sqrt(Gx**2+Gy**2)
    theta = (np.rad2deg(np.arctan2(Gy, Gx)) + 180) % 360
    #for i in range(theta.shape[0]):
    #    for j in range(theta.shape[1]):
    #        if theta[i][j]==360:
    #            theta[i][j]=0
    ### END YOUR CODE
    plt.subplot(3,2,1)
    plt.imshow(G,cmap='gray')

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)
    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    out=G.copy()
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            alpha = np.deg2rad(theta[i, j])
            # note here the angle is measured clockwisely
            # i.e. if theta=90 degree the direction is south.
            p1 = G[i - int(np.round(np.sin(alpha))), j - int(np.round(np.cos(alpha)))]
            p2 = G[i + int(np.round(np.sin(alpha))), j + int(np.round(np.cos(alpha)))]
            if not (G[i, j] >= p1 and G[i, j] >= p2):
                out[i, j] = 0
            else:
                out[i, j] = G[i, j]
    ### END YOUR CODE
    plt.subplot(3, 2, 2)
    plt.imshow(G,cmap='gray')

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges
    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    strong=img>high
    strong_edges[strong]=img[strong]
    weak = (img > low)&(img <= high)
    weak_edges[weak]=img[weak]
    ### END YOUR CODE
    plt.subplot(3,2,3)
    plt.imshow(strong_edges,cmap='gray')
    plt.subplot(3,2,4)
    plt.imshow(weak_edges,cmap='gray')

    return strong_edges, weak_edges

def get_neighbors(y, x, H, W,weak_edges,edges):
    """ Return indices of valid neighbors of (y, x)
    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)
    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                if weak_edges[i][j]!=0 and edges[i][j]==0:
                    neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    strong=strong_edges>0
    edges[strong]=1
    for y in range(H):
        for x in range(W):
            if strong_edges[y][x] != 0:
                unmarked=np.array(get_neighbors(y,x,H,W,weak_edges,edges))
                i=0
                while i < unmarked.__len__():
                    if edges[unmarked[i][0]][unmarked[i][1]] == 0 and weak_edges[unmarked[i][0]][unmarked[i][1]]!=0:
                        edges[unmarked[i][0]][unmarked[i][1]]=1
                        neighbors=get_neighbors(unmarked[i][0],unmarked[i][1],H,W,weak_edges,edges)
                        if neighbors != []:
                            unmarked=np.concatenate((unmarked,neighbors))
                    i+=1

    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.
    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE
    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.
    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.
    Args:
        img: binary image of shape (H, W)

    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H))) #得到斜对角线的长度
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)   #对应着上述公式中r的区直范围
    thetas = np.deg2rad(np.arange(-90.0, 90.0))     #对应着上述公式中角度的大小

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i, j in zip(ys, xs):    #对于zip(ys,xs)，形成了元组
        for idx in range(thetas.shape[0]):
            r = j * cos_t[idx] + i * sin_t[idx]
            accumulator[int(r + diag_len), idx] += 1    #加上diag_len的原因是因为有偏置量
    ### END YOUR CODE

    return accumulator, rhos, thetas