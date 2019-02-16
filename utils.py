import os
import cv2
import numpy as np
from pylab import imshow, imsave


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass

def crop_to_mask(image, mask, color=(0,0,0)):
    """
    Return image with background
    """
    image = crop_to_mask_color(image,mask,color)
    img_BGRA = cv2.merge((*cv2.split(image.astype(np.uint8)), mask * 255))
    return img_BGRA

def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]

def crop_to_mask_color(image, mask, color=(50,50,50)):
    """
    Return image with background
    """
    mask = np.dstack((mask, mask, mask))
    mask = mask.astype(np.uint8)
    cropped = cv2.multiply(image,mask) + abs(mask-1) * color
    return cropped

def save_out(mask_array, img, pads, output_path):
    mask_array = crop_image(mask_array, pads)
    output_image = crop_to_mask(crop_image(img, pads), (mask_array > 0.5).astype(np.uint8)).astype(np.uint8)
    #output_image = img
    imsave(output_path, output_image)
