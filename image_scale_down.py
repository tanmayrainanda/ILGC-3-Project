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
    img = cv2.imread('C:/Users/shiva/OneDrive/Documents/GitHub/ILGC-3-Project/Collected Dataset/24.jpg')
    print(img)
    scaled_img = scaleImage(img, 32, 24)
    cv2.imshow("Original Image", img)
    cv2.imshow("Scaled Image", scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()