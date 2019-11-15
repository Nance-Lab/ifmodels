import cv2
import numpy as np


def resize(fixed_image, scale_percent):
    """
    Function that resizes that universal atlas to be in proportion with input
    images.

    Parameters
    ----------
    fixed_image: array
        The atlas image of the slice that will be used for fixed
        registration coordinates.

    scale_percent: int
        The percentage of inflating the fixed image so that it is closer
        in size to moving image.

    Returns
    -------
    resized: array
        The resized atlas slice.

    """
    F_im = fixed_image.astype(np.uint16)
    width = int(F_im.shape[1] * scale_percent)
    height = int(F_im.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(F_im, dim, interpolation=cv2.INTER_AREA)
    return resized
