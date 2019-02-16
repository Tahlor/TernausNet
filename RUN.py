from pylab import imshow, imsave
import torch
from torch import nn
import numpy as np
import cv2
import utils
import os
from unet_models import unet11
from pathlib import Path
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt
import time
import multiprocessing
from utils import *

TEST=False
#print(torch.cuda.is_available())
#device = torch.device("cpu")


class Sample:

    def __init__(self, input_folder, output_folder, interval=100, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output = utils.mkdir(output_folder)
        self.interval = interval
        self.device_model = None

        poolcount = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(processes=poolcount)

        self.model = self.get_model()

        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def get_model(self):
        model = unet11(pretrained='carvana', device=self.device)
        model.eval()
        self.device_model = model.to(self.device)
        return self.device_model

    def mask_overlay(self, image, mask, color=(0, 255, 0)):
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

    def load_image(self, path, pad=True):
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

    def create_mask_from_image(self, image_path ='lexus.jpg', output_path="./lexus_crop.jpg"):
        #parent, image_name = os.path.split(image_path)
        img, pads = self.load_image(image_path, pad=True)

        if not TEST:
            with torch.no_grad():
                input_img = torch.unsqueeze(self.img_transform(img).to(self.device), dim=0)

            with torch.no_grad():
                mask = torch.sigmoid(self.model(input_img))

            mask_array = mask.data[0].cpu().numpy()[0]
        else:
            mask_array = (img * 0)[:,:,0]
            mask_array[300:600]=1

        self.pool.apply_async(save_out, args=(mask_array, img, pads, output_path))
        #print(handler.get())
        #save_out(mask_array, img, pads, output_path)

    def main(self):
        run_time = 0
        i = 0
        for f in os.listdir(input_folder):
            if i % self.interval == 0 and i > 0:
                print("Progress: {}".format(i))
                print("Speed: {}".format(run_time / self.interval))
                run_time = 0
            if f[-4:] == ".jpg":

                # Prep paths
                path = os.path.join(input_folder, f)
                output_path = os.path.join(output_folder, f).replace(".jpg", ".png")

                if not os.path.exists(output_path) or TEST:
                    i += 1
                    tic = time.clock()
                    self.create_mask_from_image(path, output_path)
                    toc = time.clock()
                    # print("Time: {}, Device: {}".format(toc-tic, device))
                    run_time += toc - tic
        self.pool.close()

if __name__ == "__main__":
    input_folder = r"../data/carvana/test" if not TEST else "."
    output_folder = r"../data/carvana/masked_images2"

    s = Sample(input_folder, output_folder)
    s.main()
    #for dir,sub,f in os.walk("../data/carvana/test"):
