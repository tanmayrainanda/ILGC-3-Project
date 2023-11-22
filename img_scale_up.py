import cv2
import numpy as np

# upscale the image and make it sharper

def upscale_image(image, scale_percent = 3000):
    # upscale the image
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    dim = (width, height)

    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # sharpen the image
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    image = cv2.filter2D(image, -1, kernel_sharpening)
    return image

cv2.imshow("upscaled",upscale_image(cv2.imread('Screenshot 2023-11-21 161538_smaller.png'),2000))
cv2.waitKey(0)