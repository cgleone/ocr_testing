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
    kernel_options = [3, 5, 7, 9]
    #rescale_options = [1, 2, 3, 4]
    rescale_options = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    lev_accuracies = []
    seq_accuracies = []
    seq_for_max_seq = 0
    lev_for_max_seq = 0
    seq_for_max_lev = 0
    lev_for_max_lev = 0
    for kernel in kernel_options:
        for rescale in rescale_options:
            seq, lev = get_text(file, kernel, rescale)
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







