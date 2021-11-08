from difflib import SequenceMatcher
import Levenshtein as lev
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os
import cv2

import preprocessing
from main import get_text
from matplotlib import pyplot as plt


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
        new_img = ImageOps.expand(image, border=(200, 200, 200, 200), fill="white")
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


def compare_kernels(file):

    value_dict = TestValueDict()

    lev_accuracies = []
    seq_accuracies = []
    seq_for_max_seq = 0
    lev_for_max_seq = 0
    seq_for_max_lev = 0
    lev_for_max_lev = 0
    for rescale in value_dict.get_value_list('rescale'):
        for kernel in value_dict.get_value_list('gaussian'):
            seq, lev = get_text(file, value_dict)
            lev_accuracies.append(lev*100)
            seq_accuracies.append(seq*100)
            preprocessing.method_list.clear()

            if seq > seq_for_max_seq:
                seq_for_max_seq = seq
                lev_for_max_seq = lev
                best_seq_kernel = kernel
                best_seq_rescale = rescale
            if lev > lev_for_max_lev:
                lev_for_max_lev = lev
                seq_for_max_lev = seq
                best_lev_kernel = kernel
                best_lev_rescale = rescale

            value_dict.inc_location_in_list('gaussian')

        value_dict.inc_location_in_list('rescale')
        value_dict.reset_location_in_list('gaussian')

    print("Best seq kernel {} and Best seq rescale {} produces accuracy of {} (seq) and {} (lev)".format(
        best_seq_kernel, best_seq_rescale, seq_for_max_seq, lev_for_max_seq))

    print("Best lev kernel {} and Best lev rescale {} produces accuracy of {} (seq) and {} (lev)".format(
        best_lev_kernel, best_lev_rescale, seq_for_max_lev, lev_for_max_lev))


    #plot_modelling(options, seq_accuracies, lev_accuracies)


def plot_modelling(kernels, seq, lev):
    plt.plot(kernels, seq, label='SequenceMatcher Accuracy')
    plt.plot(kernels, lev, label='Levenshtein Accuracy')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Rescale Factor ")
    plt.title("Comparing Rescaling Factor for Preprocessing Realistic Report 1")
    plt.legend()
    plt.show()



class TestValueDict:
    def __init__(self):

        self.set_value_lists()
        self.location_in_lists = {'gaussian': 0, 'rescale': 0, 'brightness': 0, 'contrast': 0, 'sharpen': 0}

    def set_value_lists(self):
        self.value_dict = {'gaussian': [3, 5, 7, 9],
                       'rescale': [1, 2, 3, 4],
                       'brightness': [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3],
                       'contrast': [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3],
                       'sharpen': [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3],
                       }


#[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],
    def get_value_list(self, value_type):
        return self.value_dict[value_type]

    def inc_location_in_list(self, value_type):
        self.location_in_lists[value_type] = self.location_in_lists[value_type] + 1

    def reset_location_in_list(self, value_type):
        self.location_in_lists[value_type] = 0

    def get_value(self, value_type):
        location = self.location_in_lists[value_type]
        return self.value_dict[value_type][location]



