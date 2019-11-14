# operating system dependent functionality
import os

#allows for all the pathnames of a specified pattern to be located
from glob import glob

#Package imports
import numpy as np
from PIL import Image
import nibabel as nib
import nrrd
import pandas as pd
from skimage import io
import cv2
from skimage import filters
import matplotlib.pyplot as plt


#Skimage imports
from skimage import io
from skimage import filters
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import  closing, square, remove_small_objects
from skimage.color import label2rgb
from skimage.transform import rescale, resize

def im_read(file_name):
    """
    Function that reads the image into a Jupyter Notebook and gets the max intensity
    projection.

    Parameters
    ----------
    file_name: string
        The actual name of the file that is being accessed. 

    Returns
    -------
    im_max: array
        The maximum intensity projection of the read image. 
    
    """
    im = io.imread(file_name)
    im_max = np.max(im, axis=0)
    return im_max

def mim_edge_detector(max_ip):
    """
    Function that performs the edge detection to get registration points for moving
    iamges.

    Parameters
    ----------
    max_ip: array
        The maximum intensity projection of an immunofluorescent slice image.

    Returns
    -------
    binary: array
        The maximum intensity projection of the read image. 
    
    """
    gauss = filters.gaussian(max_ip, sigma=11, output=None, mode='nearest', cval=0, 
                             multichannel=None, preserve_range=False, truncate=4.0)
    edge_sobel = filters.sobel(gauss)
    threshold = filters.threshold_otsu(edge_sobel)
    binary = edge_sobel > threshold
    return binary

def image_cleaning(binary):
    """
    A function that cleans up the image by removing small artifacts caused by 
    methodology.

    Parameters
    ----------
    max_ip: array
        The maximum intensity projection of an immunofluorescent slice image.

    Returns
    -------
    binary: array
        The maximum intensity projection of the read image. 
    
    """
    #Cleans up the image
    binary = remove_small_objects(binary, min_size=3000, connectivity=1, in_place=True)
    imagex,imagey = binary.shape
    xrange=int(imagex/10)
    yrange=int(imagey/10)
    xfull=imagex
    yfull=imagey

    return binary

def atlas_slice(atlas, slice_number):
#Importing the proper atlas slice for registration
    epi_img_data2 = atlas.get_fdata()
    sagittal = epi_img_data2[140, :, : ]
    coronal = epi_img_data2[:, slice_number, :]
    horizontal = epi_img_data2[:, : , 100]
    return sagittal, coronal, horizontal

#from: https://nipy.org/nibabel/coordinate_systems.html
def show_slices(slices):
#Allows for slicecs from .nii files to be viewed
    fig,axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap='gray', origin='lower')
    return

def nrrd_to_nii(file):
#Converting the .nrrd atlas to a .nii file format
    _nrrd = nrrd.read(file)
    data = _nrrd[0]
    header = _nrrd[1]
    F_im_nii = nib.Nifti2Image(data,np.eye(4))
    return F_im_nii

def atlas_edge_detection(image):
#Detecting the edges of the atlas image
    gauss = filters.gaussian(resized, sigma=11, output=None, mode='nearest', cval=0, 
                             multichannel=None, preserve_range=False, truncate=4.0)
    edge_sobel = filters.sobel(gauss)
    threshold = filters.threshold_otsu(edge_sobel)
    binary = edge_sobel > threshold
    return binary


def x_value(binary_image):
    #Trying to find the relative maximum at the top of the slice in a new way
    for x in range(binary_image.shape[0]):
        unique_array = np.unique(binary_image[x], axis=0)
        if unique_array.shape[0] == 2:
            break
    return x

def y_values(binary_image):
    x = x_value(binary_image)
    y_list = []
    for y in range(binary_image[x].size):
        image = binary_image[x]
        value = image[y]
        if value == True:
            y_list.append (y)
        else:
            pass
    y_list = np.array(y_list)
    return y_list

def point_middle(binary_image):
    x = x_value(binary_image)
    y = y_values(binary_image)
    middle = np.median(y)
    midpoint = int(middle)
    #midpoint = middle
    return midpoint

def local_max(binary_image):
    x = x_value(binary_image)
    y = point_middle(binary_image)
    return x, y

def minx_value(binary_image):
    #Trying to find the relative maximum at the top of the slice in a new way
    xlist = reversed(range(binary_image.shape[0]))
    for x in xlist:
        unique_array = np.unique(binary_image[x], axis=0)
        if unique_array.shape[0] == 2:
            break
    return x

def miny_values(binary_image):
    x = minx_value(binary_image)
    y_list = []
    for y in range(binary_image[x].size):
        image = binary_image[x]
        value = image[y]
        if value == True:
            y_list.append (y)
        else:
            pass
    y_list = np.array(y_list)
    return y_list

def min_middle(binary_image):
    x = minx_value(binary_image)
    y = miny_values(binary_image)
    middle = np.median(y)
    midpoint = int(middle)
    #midpoint = middle
    return midpoint

def local_min(binary_image):
    x = minx_value(binary_image)
    y = min_middle(binary_image)
    return x, y

def find_points(binary_image):
    binary_half = np.array_split(binary_image, 2, axis=0)
    left1=binary_half[0]
    left1=np.rot90(left1, k=1)
    left2=binary_half[1]
    left2=np.rot90(left2,k=1)

    x1, y1 = local_max(binary_image)
    x4, y4 = local_min(binary_image)
    sy2, x2 = local_max(left1)
    sy6, x6 = local_min(left1)
    y3, sx3 = local_max(left2)
    sy5, sx5 = local_min(left2)

    binaryx, binaryy = binary_image.shape
    lx, ly = left1.shape

    x2 = x2
    y2 = binaryy - sy2
    x3 = sx3 + ly
    y3 = (binaryy - y3)
    x5 = sx5 + ly
    y5 = (binaryy - sy5)
    y6 = (binaryy - sy6)

    coor = {'M_x': [x1, x2, x3, x4, x5, x6], 'M_y': [y1, y2, y3, y4, y5, y6]}
    coor_df = pd.DataFrame(coor, columns = ['M_x', 'M_y'])
    return coor_df

def red_points(checkx, checky, binary_image, checkpoints):
    binaryx, binaryy = binary_image.shape
    checkpoints [checkx, checky] = 255
    kernel = list(range(1,50))
    xvalues = list(range(checkx - 50, checkx+50))
    for x in xvalues:
        for distances in kernel:
            checkpoints[x, (checky - distances):(checky+distances), 0:1] = 255
    return checkpoints

def reg_coefficients(df, point1, point2, point3):
    #Getting the registration coefficients
    row1 = point1-1
    row2 = point2-1
    row3 = point3-1
    X = np.array([[df.iloc[row1][0], df.iloc[row1][1], 1], [df.iloc[row2][0], df.iloc[row2][1], 1], [df.iloc[row3][0], df.iloc[row3][1], 1]])
    X_prime = np.array([df.iloc[row1][2], df.iloc[row2][2], df.iloc[row3][2]])
    Y_prime =  np.array([df.iloc[row1][3], df.iloc[row2][3], df.iloc[row3][3]])
    a,b,c = np.linalg.solve(X, X_prime)
    d,e,f = np.linalg.solve(X,Y_prime)
    ax = np.array([[a, b, c], [d, e, f], [0, 0, 1]])
    ainv = np.linalg.inv(ax)

    return ainv

def registration(image, coefficients):
    #Performing the tranformation
    inva = coefficients[0][0]
    invb = coefficients[0][1]
    invc = coefficients[0][2]
    invd = coefficients[1][0]
    inve = coefficients[1][1]
    invf = coefficients[1][2]
    im = Image.fromarray(image)
    atlas_size = (3200, 4280)
    im12 = im.transform(atlas_size, Image.AFFINE, (inva, invb, invc, invd, inve, invf), resample=Image.NEAREST)
    registered_im = np.array(im12)

    return registered_im
