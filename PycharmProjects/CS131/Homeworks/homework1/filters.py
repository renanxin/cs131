import numpy as np
def conv_nested(image,kernel):
    """A naive implementation of convolution filter.
    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)
    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi,Wi=image.shape
    Hk,Wk=kernel.shape
    out=np.zeros((Hi,Wi))

    ### YOUR CODE HERE
    x=(Hk-1)//2
    y=(Wk-1)//2
    for a in range(x,Hi-x):
        for b in range(y,Wi-y):
            for i in range(0,Hk):
                for j in range(0,Wk):
                    out[a,b]+=kernel[i][j]*image[a-(i-x)][b-(j-y)]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.
    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)
    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)
    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out=np.zeros((H+2*pad_height,W+2*pad_width))
    out[pad_height:pad_height+H,pad_width:pad_width+W]=image[:,:]
    ### END YOUR CODE
    return out

def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.
    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.
    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)
    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    x = (Hk - 1) // 2
    y = (Wk - 1) // 2
    kernel_new=np.flip(np.flip(kernel,0),1).copy()    #Flip an array vertically
    image_padding=zero_pad(image,x,y)
    for i in range(x,Hi-x):
        for j in range(y,Wi-y):
            out[i, j] = np.sum(image[i - x:i + x + 1, j - y:j + y + 1] * kernel_new)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g
    Hint: use the conv_fast function defined above.
    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)
    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = g[1:, :]
    g_new = np.flip(np.flip(g, 1), 0).copy()
    out = conv_fast(f, g_new)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g
    Subtract the mean of g from g so that its mean becomes zero
    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)
    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g_new=g.copy()-g.mean()
    f_new=f.copy()-f.mean()
    out=cross_correlation(f_new,g_new)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g
    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.
    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)
    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    def conv_fast_new(image,kernel):
        Hi,Wi=image.shape
        Hk,Wk=kernel.shape
        out=np.zeros((Hi,Wi))
        kernel_new=np.flip(np.flip(kernel,0),1).copy()
        x=(Hk-1)//2
        y=(Wk-1)//2
        for i in range(x, Hi - x, 1):
            for j in range(y, Wi - y, 1):
                out[i, j] = np.sum(image[i - x:i + x + 1, j - y:j + y + 1] * kernel_new / image[i - x:i + x + 1,j-y:j+y+1].std()/kernel_new.std())
        return out

    def cross_correlation_new(f,g):
        out=None
        g_new=g[1:,:].copy()
        g_new=np.flip(np.flip(g_new,1),0)
        out=conv_fast_new(f,g_new)
        return out

    g_new = g.copy() - g.mean()
    f_new = f.copy() - f.mean()
    out = cross_correlation_new(f_new, g_new)
    ### END YOUR CODE

    return out
