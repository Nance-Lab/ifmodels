import cv2
import numpy as np

def resize (fixed_image, scale_percent):
    """
    Function that resizes that universal atlas to be in proportion with input
    images.

    Parameters
    ----------
    fixed_image: 
        If this is a DataFrame, it should have the columns `contrast1` and
        `answer` from which the dependent and independent variables will be
        extracted. If this is a string, it should be the full path to a csv
        file that contains data that can be read into a DataFrame with this
        specification.

    Returns
    -------
    x : array
        The unique contrast differences.
    y : array
        The proportion of '2' answers in each contrast difference
    n : array
        The number of trials in each x,y condition
    """
    F_im = fixed_image.astype(np.uint16)
    width = int(F_im.shape[1] * scale_percent)
    height = int(F_im.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(F_im, dim, interpolation = cv2.INTER_AREA)
    return resized
