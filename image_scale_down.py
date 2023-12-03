import cv2

def scaleImage(img, width, height):
    original_width = img.shape[1]
    original_height = img.shape[0]
    if original_width / original_height > width / height:
        new_width = width
        new_height = int(original_height * (width / original_width))
    else:
        new_height = height
        new_width = int(original_width * (height / original_height))
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled_img

def main():
    for i in range(0, 414):
        img = cv2.imread(f'/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset/{i}.jpg')
        if img is None:
            continue
        scaled_img = scaleImage(img, 32, 24)
        cv2.imwrite(f'/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset/scaled_down/{i}.jpg', scaled_img)
        

if __name__ == "__main__":
    main()