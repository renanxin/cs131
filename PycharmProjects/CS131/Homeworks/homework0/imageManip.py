import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

image1_path='image1.jpg'
image2_path='image2.jpg'
#############################################################################################(Question 2.1)
def load(image_path):
    """ Loads an image from a file path
    Args:
        image_path: file path to the image
    Returns:
        out: numpy array of shape(image_height, image_width, 3)     注意这里得到的是长宽高
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out=io.imread(image_path)
    ### END YOUR CODE

    return out
def display(img):
    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
image1 = load(image1_path)
image2 = load(image2_path)

#display(image1)
#display(image2)
#############################################################################################(Question 2.2)
def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2
        where x_n is the new value and x_p is the original value
    Args:
        image: numpy array of shape(image_height, image_width, 3)
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out=0.5*image**2
    #a=np.max(out)
    #out=out/(a/255)
    #out=out.astype(image.dtype)
    ### END YOUR CODE

    return out
#new_image = change_value(image1)
#display(new_image)
#############################################################################################(Question 2.3)
def convert_to_grey_scale(image):
    """ Change image to gray scale
    Args:
        image: numpy array of shape(image_height, image_width, 3)
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    out=color.rgb2gray(image)
    ### END YOUR CODE

    return out
#grey_image = convert_to_grey_scale(image1)
#display(grey_image)
#############################################################################################(Question 2.4)
def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified
    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out=image.copy()
    if channel=='R':
        out[:,:,0]=0
    elif channel=='G':
        out[:,:,1]=0
    elif channel=='B':
        out[:,:,2]=0
    ### END YOUR CODE
    return out
#without_red = rgb_decomposition(image1, 'R')
#without_blue = rgb_decomposition(image1, 'B')
#without_green = rgb_decomposition(image1, 'G')

#display(without_red)
#display(without_blue)
#display(without_green)
#############################################################################################(Question 2.5)
def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified
    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    out=lab.copy()
    out=out.astype(image.dtype)
    if channel=='L':
        out=out[:,:,0]
    elif channel=='A':
        out=out[:,:,1]
    elif channel=='B':
        out=out[:,:,2]
    ### END YOUR CODE

    return out
#image_l = lab_decomposition(image1, 'L')
#image_a = lab_decomposition(image1, 'A')
#image_b = lab_decomposition(image1, 'B')

#display(image_l)
#display(image_a)
#display(image_b)
#############################################################################################(Question 2.6)
def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified
    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    out=hsv.copy()
    out=out.astype(image.dtype)
    if channel == 'H':
        out = out[:, :, 0]
    elif channel == 'S':
        out = out[:, :, 1]
    elif channel == 'V':
        out = out[:, :, 2]
    ### END YOUR CODE

    return out
#image_h = hsv_decomposition(image1, 'H')
#image_s = hsv_decomposition(image1, 'S')
#image_v = hsv_decomposition(image1, 'V')

#display(image_h)
#display(image_s)
#display(image_v)
#############################################################################################(Question 2.7)
def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image
    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    out=np.zeros(image1.shape)
    image1=rgb_decomposition(image1,channel1)
    iamge2=rgb_decomposition(image2,channel2)
    height,width,channels=image1.shape
    out[:,0:width//2,:]=image1[:,0:width//2,:]
    out[:,width//2:,:]=image2[:,width//2:,:]
    out=out.astype(image1.dtype)
    ### END YOUR CODE

    return out
#image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
#display(image_mixed)