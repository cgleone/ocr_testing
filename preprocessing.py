import numpy as np
import imutils
import cv2
import pytesseract
import re

method_list = []

def deskew(img):
    osd = pytesseract.image_to_osd(img)
    angle = float(re.search('(?<=Rotate: )\d+', osd).group(0))
    print("angle: ", angle)

    method_list.append("Deskew from an original angle of {}".format(angle))

    if angle:
        img = imutils.rotate(img, angle=-angle)
    return img


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    method_list.append("Dilation")
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    method_list.append("Erosion")
    return cv2.erode(image, kernel, iterations=1)

#canny edge detection
def canny(image):
    method_list.append("Canny")
    return cv2.Canny(image, 100, 200)

#rescaling
def rescale(img, xfactor, yfactor):
    new_img = cv2.resize(img, None, fx=xfactor, fy=yfactor, interpolation=cv2.INTER_CUBIC)
    method_list.append("Rescaling  by fx={} and fy={}".format(xfactor, yfactor))
    return new_img
