from pylab import imshow, imsave
import torch
from torch import nn
import numpy as np
import cv2
import utils
import os
from unet_models import unet11
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def get_model():
    model = unet11(pretrained='carvana', device=device)
    model.eval()
    return model.to(device)

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def crop_to_mask(image, mask, color=(50,50,50)):
    """
    Return image with background
    """
    mask = np.dstack((mask, mask, mask))
    mask = mask.astype(np.uint8)
    inverted_mask = abs(mask-1) # invert mask

    #weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    weighted_sum = cv2.multiply(image,mask) + inverted_mask * np.array(color)

    # img = image.copy()
    # ind = mask[:, :, 1] > 0
    # img[ind] = weighted_sum[ind]
    return weighted_sum

def load_image(path, pad=True):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not pad:
        return img
    
    height, width, _ = img.shape
    
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
        
    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad
    
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2] 
    
    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]

model = get_model()

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_mask_from_image(image_path ='lexus.jpg', output_folder="./output"):
    parent, image_name = os.path.split(image_path)
    img, pads = load_image(image_path, pad=True)

    if True:
        with torch.no_grad():
            input_img = torch.unsqueeze(img_transform(img).to(device), dim=0)

        with torch.no_grad():
            mask = F.sigmoid(model(input_img))

        mask_array = mask.data[0].cpu().numpy()[0]
    else:
        mask_array = (img * 0)[:,:,0]
        mask_array[300:600]=1

    mask_array = crop_image(mask_array, pads)
    output_image = crop_to_mask(crop_image(img, pads),  (mask_array > 0.5).astype(np.uint8)).astype(np.uint8)

    # Show image
    #plt.imshow(mask_array)
    #imshow(mask_overlay(crop_image(img, pads), (mask_array > 0.5).astype(np.uint8)))

    # Save
    output_path = os.path.join(output_folder, image_name)
    #print(output_path)
    imsave(output_path, output_image)

if __name__ == "__main__":
    input_path = r"../data/carvana/test"
    #input_path = "."
    output_path = r"../carvana/masked_images"
    output = utils.mkdir(output_path)

    #for dir,sub,f in os.walk("../data/carvana/test"):
    for i,f in enumerate(os.listdir(input_path)):
        if i % 100 == 0:
            print(i)
        if f[-4:]==".jpg":
            path = os.path.join(input_path, f)
            create_mask_from_image(path,output_path)
