import cv2
import numpy as np

# read all images for folder name abc and flip them horizontally and save them in the same folder

folder_name = 'Collected Dataset'
for i in range(135,208):
    try:
        img = cv2.imread(f'{folder_name}/{i}.jpg')
    except:
        FileNotFoundError
    img = cv2.flip(img, 1)
    cv2.imwrite(f'Collected Dataset/flipped/{i}_flipped.jpg', img)