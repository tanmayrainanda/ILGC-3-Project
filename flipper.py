import cv2
import numpy as np

# read all images for folder name abc and flip them horizontally and save them in the same folder

folder_name = 'Collected Dataset'
for i in range(267,280):
    try:
        img = cv2.imread(f'{folder_name}/{i}.jpg')
        img = cv2.flip(img, 0)
        cv2.imwrite(f'Collected Dataset/flipped/{i+161}.jpg', img)
    except:
        FileNotFoundError