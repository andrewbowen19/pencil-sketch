# Trying out the pencil-sketch snippet we found online
# Thank you, LinkedIn: https://www.linkedin.com/search/results/content/?keywords=convert%20picture%20to%20sketch%20using%20python&sid=FAM&update=urn%3Ali%3Afs_updateV2%3A(urn%3Ali%3Aactivity%3A6936627290571194368%2CBLENDED_SEARCH_FEED%2CEMPTY%2CDEFAULT%2Cfalse)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

class PencilSketch:

    def __init__(self):
        image_name: str

    def create_pencil_sketch(self, image_path: str, ksize:int = 21, sigma_x: int = 0, sigma_y: int = 0) -> (np.array, np.array):
        '''
        Create pencil sketch drawing version of an image
        
        parameters:
            image_path : str; file name/path to image to be converted, should be in jpg format
            ksize: int; Size of kernel for Gaussian Blur applied by cv2 (must be odd)
            sigma_x, sigma_y: int;  Standard deviation value of kernal along horizontal/vertical direction.
        '''
        # Read image and remove color dimension
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray_image

        # Appling gaussian blue to inverted image 
        blur = cv2.GaussianBlur(inverted, ksize=(ksize, ksize), sigmaX=sigma_x, sigmaY=sigma_y)
        inverted_blur = 255 - blur
        sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)

        return gray_image, sketch


if __name__=="__main__":

    # kernels = {
    #            "jeff-laurel-sunset.jpg":81,
    #            "milllie-hardwood.jpg":51,
    #            "millie-boat.jpg":15,
    #            "millie-cute-as-a-button.jpg":81,
    #            "zach-balloons.jpg":25,
    #            "kristine-amelia-park.jpg": 27,
    #            "drew-beautiful.jpg": 7
    #            }

    # for f in os.listdir("img/sample"):
    #     if ".jpg" in f:
    #         print(f)
    #         im_path = os.path.join("img", "sample", f)
    #         img, sketch = pencil_sketch(im_path, ksize=kernels.get(f))
            
    #         # cv2.startWindowThread()
            
    #         cv2.imshow(f, sketch)
    #         im_concat = np.concatenate((img, sketch), axis=1)
    #         compare = cv2.imshow("Side-by-side", im_concat)
    #         comparison_file_name = "pencil_compare-" + f
    #         cv2.imwrite(comparison_file_name, im_concat)

    #         cv2.waitKey(0)

    # # plt.show()

    # Getting command-line arguments 
    parser = argparse.ArgumentParser(
                                     prog= 'pencil-sketch',
                                     description = 'Python tool to render pencil drawings of images'       
                                    )
    parser.add_argument('-f', '--filename', default="img.jpg", type=str)  
    parser.add_argument('-k', '--kernel', default=21, type=int)
    parser.add_argument('-sx', '--sigmaX', default=0, type=int)  
    parser.add_argument('-sy', '--sigmaY', default=0, type=int)  

    args = parser.parse_args()
    filename = args.filename

    print(f"Filename: {args.filename}")
    print(args.kernel, args.sigmaX, args.sigmaY)
    # Creating pencil sketch
    pencil = PencilSketch()
    image, sketch = pencil.create_pencil_sketch(filename,
                                                args.kernel,
                                                args.sigmaX, args.sigmaY)

    # Display img
    cv2.imshow(filename, sketch)

    # Image stays on screen until keystroke
    cv2.waitKey(0)

    



