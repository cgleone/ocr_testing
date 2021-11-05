import numpy as np
import imutils
import cv2
import pytesseract
import re

method_list = []

def deskew_old(img):
    osd = pytesseract.image_to_osd(img)
    angle = float(re.search('(?<=Rotate: )\d+', osd).group(0))
    print("angle: ", angle)

    method_list.append("Deskew from an original angle of {}".format(angle))

    if angle:
        img = imutils.rotate(img, angle=-angle)
    return img

def deskew_test(img):

    cv2.imshow("image", img)
    cv2.waitKey(0)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.bitwise_not(gray_img)
    coordinates = np.column_stack(np.where(gray_img > 0))
    ang = cv2.minAreaRect(coordinates)[-1]
    print(ang)
    if ang < -45:
        angle = -(90 + ang)
    else:
        angle = -ang

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


# got this code from https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
def get_skew_angle(image) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    cv2.imshow("image", image)
    cv2.waitKey(0)
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

    # Find largest contour and surround in min area box

    allContourAngles = [cv2.minAreaRect(c)[-1] for c in contours]
    angle = sum(allContourAngles) / len(allContourAngles)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    # angle = minAreaRect[-1]
    # if angle < -45:
    #     angle = 90 + angle
    # return -1.0 * angle

    return angle


def rotate_image(image, angle: float):
    new_image = image.copy()
    (h, w) = new_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(new_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return new_image


def deskew(image):
    angle = get_skew_angle(image)
    print(angle)
    return rotate_image(image, -1.0 * angle)