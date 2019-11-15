# Package imports
import numpy as np
from PIL import Image
import nibabel as nib
import nrrd
import pandas as pd
from skimage import io
from skimage import filters
import matplotlib.pyplot as plt


# Skimage imports
from skimage.morphology import remove_small_objects


def im_read(file_name):
    """
    Function that reads the image into a Jupyter Notebook and gets the
    max intensity projection.

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
    Function that performs the edge detection to get registration points
    for moving images.

    Parameters
    ----------
    max_ip: array
        The maximum intensity projection of an immunofluorescent slice image.

    Returns
    -------
    binary: array
        The maximum intensity projection of the read image.

    """
    gauss = filters.gaussian(max_ip, sigma=11, output=None, mode='nearest',
                             cval=0, multichannel=None, preserve_range=False,
                             truncate=4.0)
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
    binary = remove_small_objects(binary, min_size=3000, connectivity=1,
                                  in_place=True)
    # imagex,imagey = binary.shape
    # xrange=int(imagex/10)
    # yrange=int(imagey/10)
    # xfull=imagex
    # yfull=imagey

    return binary


def atlas_slice(atlas, slice_number):
    """
    A function that pulls the data for a specific atlas slice.

    Parameters
    ----------
    atlas: nrrd
        Atlas segmentation file that has a stack of slices.

    slice_number: int
        The number in the slice that corresponds to the fixed image
        for registration.

    Returns
    -------
    sagittal: array
        Sagittal view being pulled from the atlas.

    coronal: array
        Coronal view being pulled from the atlas.

    horizontal: arrary
        Horizontal view being pulled from the atlas.

    """
    epi_img_data2 = atlas.get_fdata()
    sagittal = epi_img_data2[140, :, :]
    coronal = epi_img_data2[:, slice_number, :]
    horizontal = epi_img_data2[:, :, 100]
    return sagittal, coronal, horizontal


def show_slices(slices):
    """
    A function that allows for slices from .nii files to be viewed.

    Parameters
    ----------
    slices: tuples
        Tuple of coronal, sagittal, and horizontal slices you want to view

    Returns
    -------
    N/A: This is specifically a visualization step

    Notes
    -------
    From: #from: https://nipy.org/nibabel/coordinate_systems.html

    """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap='gray', origin='lower')
    return


def nrrd_to_nii(file):
    """
    A function that converts the .nrrd atlas to .nii file format

    Parameters
    ----------
    file: tuples
        Tuple of coronal, sagittal, and horizontal slices you want to view

    Returns
    -------
    F_im_nii: nibabel.nifti2.Nifti2Image
        A nifti file format that is used by various medical imaging techniques.

    Notes
    -------
    From: #from: https://nipy.org/nibabel/coordinate_systems.html

    """
    _nrrd = nrrd.read(file)
    data = _nrrd[0]
    header = _nrrd[1]  # noqa: F841
    F_im_nii = nib.Nifti2Image(data, np.eye(4))
    return F_im_nii


def atlas_edge_detection(image):
    """
    A function that detects the edges of the atlas function

    Parameters
    ----------
    image: array
        Array that depicts that specific atlas slice being used as a
        fixed image.

    Returns
    -------
    binary: array
        The array depicting the specific atlas as a boolean.

    """
    gauss = filters.gaussian(image, sigma=11, output=None, mode='nearest', cval=0, 
                             multichannel=None, preserve_range=False, truncate=4.0)
    edge_sobel = filters.sobel(gauss)
    threshold = filters.threshold_otsu(edge_sobel)
    binary = edge_sobel > threshold
    return binary


def x_value(binary_image):
    """
    A function that finds the x-value of the relative maxium at the top of a slice

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean.
  
    Returns
    -------
    x: int
        The x-coordinate of the maximum part of the curve.

    """
    for x in range(binary_image.shape[0]):
        unique_array = np.unique(binary_image[x], axis=0)
        if unique_array.shape[0] == 2:
            break
    return x


def y_values(binary_image):
    """
    A function that finds the y-value of the relative maxium at the top of a slice

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean
   
    Returns
    -------
    y_list: list
        A list of y-values that had a boolean true value

    """
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
    """
    A function that finds the middle point if the maximum value row has more than one true
    pixel.

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean
  
    Returns
    -------
    midpoint: int
        The middle point of a the true values at the maximum curvature of a slice.

    """
    x = x_value(binary_image)
    y = y_values(binary_image)
    middle = np.median(y)
    midpoint = int(middle)
    return midpoint


def local_max(binary_image):
    """
    A function that finds the x,y cordinates of a local max.

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean
  
    Returns
    -------
    x: int
        The x coordinate of the local max of curvature.
 
    y: int
        The y coordinate of the local max of curvature.

    Notes
    -------
    Utilizes the above equations to find very close to the actual maximum of curvature.

    """
    x = x_value(binary_image)
    y = point_middle(binary_image)
    return x, y


def minx_value(binary_image):
    """
    A function that finds the relative max of the bottom of a slice.

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean
  
    Returns
    -------
    x: int
        The x coordinate of the local max of curvature.

    """
    xlist = reversed(range(binary_image.shape[0]))
    for x in xlist:
        unique_array = np.unique(binary_image[x], axis=0)
        if unique_array.shape[0] == 2:
            break
    return x


def miny_values(binary_image):
    """
    A function that finds the relative max of the bottom of a slice.

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean

    Returns
    -------
    y_list: list
        The list of y-coordinates of the local maximum with true values.

    """
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
    """
    A function that finds the middle point of the relative max of the bottom of a slice.

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean

    Returns
    -------
    midpoint: int
        The middle value location of the min value of the bottom of the slice.

    """
    x = minx_value(binary_image)
    y = miny_values(binary_image)
    middle = np.median(y)
    midpoint = int(middle)
    return midpoint


def local_min(binary_image):
    """
    A function that finds the x and y coordinates of the relative max of the bottom of a slice.

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean
  
    Returns
    -------
    x: int
        The x location of the min value of the bottom of the slice. 
 
    y: int
        The y coordinates of the local maximum curvature at the bottom of the slice.

    """
    x = minx_value(binary_image)
    y = min_middle(binary_image)
    return x, y


def find_points(binary_image):
    """
    A function that finds all of the x,y registration coordinates for six curvatures
    on full brain slices.

    Parameters
    ----------
    binary_image: boolean array
        Array that depicts the slice as a boolean

    Returns
    -------
    coor_df: pandas dataframe
        A pandas dataframe with the x and y coordinates of all six local max curvatures.

    """
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
    """
    A function that allows for better user visualization of the registration points.

    Parameters
    ----------
    checkx: int
        x-coordinate of one of the maximum curvature points
    
    checky: int
        y-coordinate of the one of the maxiumum curvature points

    binary_image: boolean array
        Array that depicts the slice as a boolean array

    checkpoints: array
        An array of zeros the size of the fixed image that will be filled with values.

    Returns
    -------
    coor_df: checkpoints
        The input array with areas marked for the registration points.

    """
    binaryx, binaryy = binary_image.shape
    checkpoints [checkx, checky] = 255
    kernel = list(range(1,50))
    xvalues = list(range(checkx - 50, checkx+50))
    for x in xvalues:
        for distances in kernel:
            checkpoints[x, (checky - distances):(checky+distances), 0:1] = 255
    return checkpoints


def reg_coefficients(df, point1, point2, point3):
    """
    A function that gets the registration coefficients for the affine transformation.

    Parameters
    ----------
    df: pandas dataframe
        Empty dataframe to store all of the coefficients with columns 'coor_df' and
        'fim_coor_df'

    point1: int
        Registration label of your first point of choice.

    point2: int
        Registration label of your second point of choice.

    point3: int
        Registration label of yoru third point of choice.

    Returns
    -------
    ainv: array
        An array that has the inverse of your coefficients from your affien transformation. 

    """
    row1 = point1-1
    row2 = point2-1
    row3 = point3-1
    X = np.array([[df.iloc[row1][0], df.iloc[row1][1], 1], [df.iloc[row2][0], df.iloc[row2][1], 1], 
                  [df.iloc[row3][0], df.iloc[row3][1], 1]])
    X_prime = np.array([df.iloc[row1][2], df.iloc[row2][2], df.iloc[row3][2]])
    Y_prime =  np.array([df.iloc[row1][3], df.iloc[row2][3], df.iloc[row3][3]])
    a,b,c = np.linalg.solve(X, X_prime)
    d,e,f = np.linalg.solve(X,Y_prime)
    ax = np.array([[a, b, c], [d, e, f], [0, 0, 1]])
    ainv = np.linalg.inv(ax)

    return ainv


def registration(image, coefficients):
    """
    A function that transforms the moving image onto a fixed atlas.

    Parameters
    ----------
    image: array
        Array that contains the pixel values of your moving image.

    coefficients: array
        The array that contains the inverse coefficient for the registration.

    Returns
    -------
    registered_im: array
        An array that contains the registered image. 

    """
    #Performing the tranformation
    inva = coefficients[0][0]
    invb = coefficients[0][1]
    invc = coefficients[0][2]
    invd = coefficients[1][0]
    inve = coefficients[1][1]
    invf = coefficients[1][2]
    im = Image.fromarray(image)
    atlas_size = (3200, 4280)
    im12 = im.transform(atlas_size, Image.AFFINE, (inva, invb, invc, invd, inve, invf), 
                        resample=Image.NEAREST)
    registered_im = np.array(im12)

    return registered_im
