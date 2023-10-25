import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess
import matplotlib.pyplot as plt

tess.pytesseract.tesseract_cmd = (r'C:\\Users\\aabha\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract')

def preprocess(img):
    imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)

    ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # row, col = 1, 3
    # fig, axs = plt.subplots(row, col, figsize=(15, 10))
    # fig.tight_layout()

    # axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # axs[0].set_title('Input')
    # cv2.imwrite('Input.jpg', img)

    # axs[1].imshow(cv2.cvtColor(sobelx, cv2.COLOR_BGR2RGB))
    # axs[1].set_title('Sobel')
    # cv2.imwrite('Sobel.jpg', sobelx)
    
    # axs[2].imshow(cv2.cvtColor(threshold_img, cv2.COLOR_BGR2RGB))
    # axs[2].set_title('Threshold')
    # cv2.imwrite('Threshold.jpg', threshold_img)

    # plt.show() 
    return threshold_img

def cleanPlate(plate): 
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thresh = cv2.dilate(gray, kernel, iterations = 1)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea, w, h):
            return plate, None

        cleaned_final = thresh[y: y + h, x: x + w]
        # plt.imshow(cv2.cvtColor(cleaned_final, cv2.COLOR_BGR2RGB))
        # plt.title('Function Test'); plt.show()
        
        return cleaned_final, [x, y, w, h]
    else:
        return plate, None

def extract_contours(threshold_img):
    element = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src = threshold_img, op = cv2.MORPH_CLOSE, kernel = element, dst = morph_img_threshold)
    
    # plt.imshow(cv2.cvtColor(morph_img_threshold, cv2.COLOR_BGR2RGB))
    # plt.title('Morphed'); plt.show()

    contours, hierarchy= cv2.findContours(morph_img_threshold, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
    return contours

def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    aspect = 4.7272
    min = 15 * aspect * 15  # minimum area
    max = 125 *aspect * 125  # maximum area
    rmin = 3
    rmax = 6

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True

def isMaxWhite(plate):
    avg = np.mean(plate)
    if(avg >= 115):
        return True
    else:
        return False

def validateRotationAndRatio(rect):
    (x, y), (width, height), rect_angle = rect

    if(width > height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle
    if angle > 15:
        return False
    if height == 0 or width == 0:
        return False
    area = height * width
    if not ratioCheck(area, width, height):
        return False
    else:
        return True

def cleanAndRead(img, contours):
    for i, cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)
        if not validateRotationAndRatio(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y: y + h , x: x + w]
            if isMaxWhite(plate_img):
                clean_plate, rect = cleanPlate(plate_img)
                if rect:
                    row, col = 1, 2
                    fig, axs = plt.subplots(row, col, figsize = (15, 10))
                    fig.tight_layout()
                    
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x + x1, y + y1, w1, h1
                    
                    axs[0].imshow(cv2.cvtColor(clean_plate, cv2.COLOR_BGR2RGB))
                    axs[0].set_title('Cleaned Plate')
                    cv2.imwrite('cleaned_plate.jpg', clean_plate)
                    #cv2.imshow("Cleaned", clean_plate)

                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang = 'eng')
                    print("Detected Text : ", text)

                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axs[1].set_title('Detected Plate')
                    cv2.imwrite('detected_plate.jpg', img)
                    
                    plt.show()

def rescaleFrame(frame):
    width = int(450)
    height = int(350)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)

if __name__ == '__main__':
    print("DETECTING PLATE . . .")

    # Path to the license plate you wish to read
    img0 = cv2.imread("car2.png")
    img = rescaleFrame(img0)
    threshold_img = preprocess(img)
    contours = extract_contours(threshold_img)
    cleanAndRead(img, contours)
    cv2.destroyAllWindows()