import numpy as np
import imutils
import cv2
import pytesseract
import re

import accuracy_testing
from PIL import Image, ImageEnhance

method_list = []

# ------------------------ preprocessing methods ---------------------------------

def deskew(image):
    angle_from_axis = get_skew_angle(image)
    if abs(angle_from_axis/90) == 1 or abs(angle_from_axis) < 1:
        axis_aligned_img = image
    else:
        axis_aligned_img = rotate(image, angle_from_axis)

    angle_by_text = get_text_rotation(axis_aligned_img)
    fully_aligned_img = rotate(axis_aligned_img, angle_by_text)

    return fully_aligned_img


def dilate(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    method_list.append("Dilation")
    return cv2.dilate(image, kernel, iterations=1)


def erode(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    method_list.append("Erosion")
    return cv2.erode(image, kernel, iterations=1)


def greyscale(img):
    method_list.append("Greyscale")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def thresholding(img):
    method_list.append("Thresholding")
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def gaussian_blur(img, kernel):
    method_list.append("Gaussian Blur with a {}x{} kernel".format(kernel, kernel))
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def median_blur(img, kernel):
    method_list.append("Median Blur")
    return cv2.medianBlur(img, kernel)


def averaging_blur(img, kernel):
    method_list.append("Averaging Blur")
    return cv2.blur(img, (kernel, kernel))


def bilateral_Filter(img):
    method_list.append("Averaging Blur")
    return cv2.bilateralFilter(img, 9, 75, 75)


def canny(image):
    method_list.append("Canny")
    return cv2.Canny(image, 100, 200)


def rescale(img, factor):
    new_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    method_list.append("Rescaling  by fx={} and fy={}".format(factor, factor))
    return new_img


def closing(img, kernel):
    img = dilate(img, kernel)
    img = erode(img, kernel)
    return img


def sharpen(pil_img, factor):
    sharp_enhancer = ImageEnhance.Sharpness(pil_img)
    new_img = sharp_enhancer.enhance(factor)
    return new_img



def brighten(pil_img, factor):
    bright_enhancer = ImageEnhance.Brightness(pil_img)
    new_img = bright_enhancer.enhance(factor)
    return new_img



def contrast(pil_img, factor):
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    new_img = contrast_enhancer.enhance(factor)
    return new_img



# def sharpen(img):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     new_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
#     return new_img


# --------------------- methods called by preprocessing methods ---------------------------------------

# got this code from https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
def get_skew_angle(image) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    #Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotate(img, angle):
    if angle:
        #widths = accuracy_testing.get_border(img, angle, format="open_cv")
        # if widths[0]:
        #     img = cv2.copyMakeBorder(img, widths[0], widths[1], widths[0], widths[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = imutils.rotate(img, angle=-angle)
    return img


def get_text_rotation(img):
    osd = pytesseract.image_to_osd(img)
    angle = float(re.search('(?<=Rotate: )\d+', osd).group(0))
    print("angle: ", angle)

    method_list.append("Deskew from an original angle of {}".format(angle))
    return angle
