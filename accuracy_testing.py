from difflib import SequenceMatcher
import Levenshtein as lev
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os
import cv2


def get_accuracies(img_name, converted_text):

    txt_filename = img_name.rstrip('.jgp') + '.txt'
    path = 'original_text/' + txt_filename

    with open(path) as f:
        actual_text = f.read()
        f.close()

    seq_match = sequence_matcher(converted_text, actual_text)
    levenshtein = levenshtein_similarity(converted_text, actual_text)

    return seq_match, levenshtein


def sequence_matcher(converted, actual):
    return SequenceMatcher(None, converted, actual).ratio()

def levenshtein_similarity(converted, actual):
    return lev.ratio(converted, actual)


def get_border(image, angle, format="pillow"):

    angle = np.abs(angle)
    if angle > 45 and angle <=90:
        angle = 90-angle

    angle_rad = np.deg2rad(angle)
    if format == "pillow":
        height = image.size[1]
    elif format == "open_cv":
        height = image.shape[0]

    vertical = int(round((height/2) - (height/2)*np.cos(angle_rad), 0))
    horizontal = int(round((height/2)*np.sin(angle_rad), 0))

    return (vertical, horizontal, vertical, horizontal)

def noisify(image, rotation=0, brightness=1, contrast=1, sharpness=1):


    if rotation:
        border = get_border(image, rotation) # top right bottom left
        new_img = ImageOps.expand(image, border=border, fill="white")
        new_img = new_img.rotate(rotation)
    else:
        new_img = image


    bright_enhancer = ImageEnhance.Brightness(new_img)
    new_img = bright_enhancer.enhance(brightness)

    contrast_enhancer = ImageEnhance.Contrast(new_img)
    new_img = contrast_enhancer.enhance(contrast)

    sharp_enhancer = ImageEnhance.Sharpness(new_img)
    new_img = sharp_enhancer.enhance(sharpness)


    return new_img





