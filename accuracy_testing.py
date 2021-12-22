from difflib import SequenceMatcher
import Levenshtein as lev
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os
import cv2
from datetime import datetime

import preprocessing
from main import get_text
from matplotlib import pyplot as plt
# from temporary_acc_total import update_file, file_date_key
import sys


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

    vertical = int(round((height/2) - (height/2)*np.cos(angle_rad), 0)) + 20
    horizontal = int(round((height/2)*np.sin(angle_rad), 0)) + 20

    return (horizontal, vertical, horizontal, vertical)

def noisify(image, rotation=0, brightness=1, contrast=1, sharpness=1):

    if rotation:
        border = get_border(image, rotation) # left top right bottom
        new_img = ImageOps.expand(image, border=border, fill="white")
        new_img = new_img.rotate(rotation, fillcolor='white')
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
    best_seq_vals = [0, 0, 0, 0, 0]
    best_lev_vals = [0, 0, 0, 0, 0]
    best_combined_vals = [0, 0, 0, 0, 0]

    seq_for_max_seq = 0
    lev_for_max_seq = 0
    seq_for_max_lev = 0
    lev_for_max_lev = 0
    seq_for_max_combined = 0
    lev_for_max_combined = 0
    best_combined_accs = 0

    count = 1
    #
    # files = os.listdir("temporary")
    # files = sorted(files, key=file_date_key)

    for rescale_val in value_dict.get_value_list('rescale'):
        for gaus_val in value_dict.get_value_list('gaussian'):
            for bright_val in value_dict.get_value_list('brighten'):
                for cont_val in value_dict.get_value_list('contrast'):
                    for sharp_val in value_dict.get_value_list('sharpen'):

                        print("\nI'm on iteration number {}\n".format(count))
                        # update_file("temporary/" + files[count-1], "sharp val: " + str(sharp_val))
                        # update_file("temporary/" + files[count-1], "contrast val: " + str(cont_val))
                        # update_file("temporary/" + files[count-1], "brightness val: " + str(bright_val))
                        #
                        count = count + 1


                        seq, lev = get_text(file, value_dict)
                        lev_accuracies.append(lev * 100)
                        seq_accuracies.append(seq * 100)
                        preprocessing.method_list.clear()

                        if seq > seq_for_max_seq:
                            seq_for_max_seq = seq
                            lev_for_max_seq = lev
                            best_seq_vals = [sharp_val, cont_val, bright_val, gaus_val, rescale_val]
                        if lev > lev_for_max_lev:
                            lev_for_max_lev = lev
                            seq_for_max_lev = seq
                            best_lev_vals = [sharp_val, cont_val, bright_val, gaus_val, rescale_val]

                        combined_accs = seq + lev
                        if combined_accs > best_combined_accs:
                            seq_for_max_combined = seq
                            lev_for_max_combined = lev
                            best_combined_accs = combined_accs
                            best_combined_vals = [sharp_val, cont_val, bright_val, gaus_val, rescale_val]
                        combined_accs = 0

                        value_dict.inc_location_in_list('sharpen')

                    value_dict.reset_location_in_list('sharpen')
                    value_dict.inc_location_in_list('contrast')

                value_dict.reset_location_in_list('contrast')
                value_dict.inc_location_in_list('brighten')

            value_dict.reset_location_in_list('brighten')
            value_dict.inc_location_in_list('gaussian')

        value_dict.reset_location_in_list('gaussian')
        value_dict.inc_location_in_list('rescale')


    print("\n\n----------------------------------------------------------------\n\n")

    print("Best seq accuracy was {} ".format(seq_for_max_seq*100))
    print("Corresponting lev accuracy was {}".format(lev_for_max_seq*100))
    print("The values that provided this maximum were: ")
    print("Sharpen Factor {}".format(best_seq_vals[0]))
    print("Contrast Factor {}".format(best_seq_vals[1]))
    print("Brighten Factor {}".format(best_seq_vals[2]))
    print("Gaussian Kernel {}".format(best_seq_vals[3]))
    print("Rescale Factor {}".format(best_seq_vals[4]))

    print("\n\n----------------------------------------------------------------\n\n")

    print("Best lev accuracy was {} ".format(lev_for_max_lev * 100))
    print("Corresponting seq accuracy was {}".format(seq_for_max_lev * 100))
    print("The values that provided this maximum were: ")
    print("Sharpen Factor {}".format(best_lev_vals[0]))
    print("Contrast Factor {}".format(best_lev_vals[1]))
    print("Brighten Factor {}".format(best_lev_vals[2]))
    print("Gaussian Kernel {}".format(best_lev_vals[3]))
    print("Rescale Factor {}".format(best_lev_vals[4]))

    print("\n\n----------------------------------------------------------------\n\n")

    print("Best combined accuracy was ".format(best_combined_accs*100))
    print("Corresponting lev accuracy was {}".format(lev_for_max_combined*100))
    print("Corresponting seq accuracy was {}".format(seq_for_max_combined*100))
    print("The values that provided this maximum were: ")
    print("Sharpen Factor {}".format(best_combined_vals[0]))
    print("Contrast Factor {}".format(best_combined_vals[1]))
    print("Brighten Factor {}".format(best_combined_vals[2]))
    print("Gaussian Kernel {}".format(best_combined_vals[3]))
    print("Rescale Factor {}".format(best_combined_vals[4]))

    print("\n\n----------------------------------------------------------------\n\n")

    write_results_file(value_dict.value_dict, file,
                       seq_for_max_seq, lev_for_max_seq, best_seq_vals,
                       lev_for_max_lev, seq_for_max_lev, best_lev_vals,
                       best_combined_accs, lev_for_max_combined, seq_for_max_combined, best_combined_vals)



def two_methods(file):
    value_dict = TestValueDict()

    lev_accuracies = []
    seq_accuracies = []
    best_seq_vals = [0, 0]
    best_lev_vals = [0, 0]

    seq_for_max_seq = 0
    lev_for_max_seq = 0
    seq_for_max_lev = 0
    lev_for_max_lev = 0

    for gauss_val in value_dict.get_value_list('gaussian'):
        for closing_val in value_dict.get_value_list('closing'):

            seq, lev = get_text(file, value_dict)
            lev_accuracies.append(lev * 100)
            seq_accuracies.append(seq * 100)
            preprocessing.method_list.clear()

            if seq > seq_for_max_seq:
                seq_for_max_seq = seq
                lev_for_max_seq = lev
                best_seq_vals = [gauss_val, closing_val]
            if lev > lev_for_max_lev:
                lev_for_max_lev = lev
                seq_for_max_lev = seq
                best_lev_vals = [gauss_val, closing_val]

            value_dict.inc_location_in_list('closing')

        value_dict.reset_location_in_list('closing')
        value_dict.inc_location_in_list('gaussian')

    print("\n\n----------------------------------------------------------------\n\n")

    print("Best seq accuracy was {} ".format(seq_for_max_seq * 100))
    print("Corresponting lev accuracy was {}".format(lev_for_max_seq * 100))
    print("The values that provided this maximum were: ")
    print("Gaussian Kernel {}".format(best_seq_vals[0]))
    print("Closing Kernal {}".format(best_seq_vals[1]))

    print("\n\n----------------------------------------------------------------\n\n")

    print("Best lev accuracy was {} ".format(lev_for_max_lev * 100))
    print("Corresponting seq accuracy was {}".format(seq_for_max_lev * 100))
    print("The values that provided this maximum were: ")
    print("Gaussian Kernel {}".format(best_lev_vals[0]))
    print("Closing Kernal {}".format(best_lev_vals[1]))


    print("\n\n----------------------------------------------------------------\n\n")




def plot_modelling(kernels, seq, lev):
    plt.plot(kernels, seq, label='SequenceMatcher Accuracy')
    plt.plot(kernels, lev, label='Levenshtein Accuracy')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Gaussian Kernel ")
    plt.title("Comparing Gaussian Factor for Preprocessing Realistic Report 1")
    plt.legend()
    plt.show()

def optimize_single_method(file):

    value_dict = TestValueDict()

    lev_accuracies = []
    seq_accuracies = []

    for kernel in value_dict.get_value_list('median'):
        seq, lev = get_text(file, value_dict)
        lev_accuracies.append(lev * 100)
        seq_accuracies.append(seq * 100)
        preprocessing.method_list.clear()
        value_dict.inc_location_in_list('median')


    rounded_lev = []
    rounded_seq = []

    for acc in lev_accuracies:
        rounded_lev.append(round(acc, 2))

    for acc in seq_accuracies:
        rounded_seq.append(round(acc, 2))

    print("Lev accuracies: {}".format(rounded_lev))
    print("Seq accuracies: {}".format(rounded_seq))

    plt.plot(value_dict.get_value_list('median'), lev_accuracies, label='Levenshtein Accuracy')
    plt.plot(value_dict.get_value_list('median'), seq_accuracies, label='Sequence Matcher Accuracy')
    plt.title('OCR Accuracy vs Median Kernel Size')
    plt.xlabel('Median Kernel Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('Modeling_graphs/11-28 Median_Anon4.jpg')
    # plot_modelling(value_dict.get_value_list('gaussian'), seq_accuracies, lev_accuracies)


def optimized_values_test(file):

    value_dict = TestValueDict()

    lev_accuracies = []
    seq_accuracies = []

    for kernel in value_dict.get_value_list('rescale'):
        seq, lev = get_text(file, value_dict)
        lev_accuracies.append(lev * 100)
        seq_accuracies.append(seq * 100)
        preprocessing.method_list.clear()
        value_dict.inc_location_in_list('rescale')
        value_dict.inc_location_in_list('sharpen')
        value_dict.inc_location_in_list('brighten')
        value_dict.inc_location_in_list('contrast')
        value_dict.inc_location_in_list('gaussian')


    rounded_lev = []
    rounded_seq = []

    for acc in lev_accuracies:
        rounded_lev.append(round(acc, 2))

    for acc in seq_accuracies:
        rounded_seq.append(round(acc, 2))

    print("Lev accuracies: {}".format(rounded_lev))
    print("Seq accuracies: {}".format(rounded_seq))


def write_results_file(value_dict, image_name,
                       seq_for_max_seq, lev_for_max_seq, best_seq_vals,
                       lev_for_max_lev, seq_for_max_lev, best_lev_vals,
                       best_combined_accs, lev_for_max_combined, seq_for_max_combined, best_combined_vals):

    now = datetime.now()

    # dd/mm/YY H:M:S
    dt = now.strftime("_%d_%m_%Y_%H-%M-%S")
    file = open('modelling_results/' + dt + '.txt', "w+")

    file.write('This test was run on the file: {}\n'.format(image_name))

    file.write('\nThe options for each processing method were:\n'.format(image_name))

    for method in value_dict.keys():
        file.write("{}:  {}\n".format(method, value_dict[method]))


    file.write("\n\n\n----------------------------------------------------------------\n\n")

    file.write("\nBest seq accuracy was {} ".format(seq_for_max_seq * 100))
    file.write("\nCorresponting lev accuracy was {}".format(lev_for_max_seq * 100))
    file.write("\nThe values that provided this maximum were: ")
    file.write("\nSharpen Factor {}".format(best_seq_vals[0]))
    file.write("\nContrast Factor {}".format(best_seq_vals[1]))
    file.write("\nBrighten Factor {}".format(best_seq_vals[2]))
    file.write("\nGaussian Kernel {}".format(best_seq_vals[3]))
    file.write("\nRescale Factor {}".format(best_seq_vals[4]))

    file.write("\n\n\n----------------------------------------------------------------\n\n")

    file.write("\nBest lev accuracy was {} ".format(lev_for_max_lev * 100))
    file.write("\nCorresponting seq accuracy was {}".format(seq_for_max_lev * 100))
    file.write("\nThe values that provided this maximum were: ")
    file.write("\nSharpen Factor {}".format(best_lev_vals[0]))
    file.write("\nContrast Factor {}".format(best_lev_vals[1]))
    file.write("\nBrighten Factor {}".format(best_lev_vals[2]))
    file.write("\nGaussian Kernel {}".format(best_lev_vals[3]))
    file.write("\nRescale Factor {}".format(best_lev_vals[4]))

    file.write("\n\n\n----------------------------------------------------------------\n\n")

    file.write("\nBest combined accuracy was ".format(best_combined_accs * 100))
    file.write("\nCorresponting lev accuracy was {}".format(lev_for_max_combined * 100))
    file.write("\nCorresponting seq accuracy was {}".format(seq_for_max_combined * 100))
    file.write("\nThe values that provided this maximum were: ")
    file.write("\nSharpen Factor {}".format(best_combined_vals[0]))
    file.write("\nContrast Factor {}".format(best_combined_vals[1]))
    file.write("\nBrighten Factor {}".format(best_combined_vals[2]))
    file.write("\nGaussian Kernel {}".format(best_combined_vals[3]))
    file.write("\nRescale Factor {}".format(best_combined_vals[4]))

    file.write("\n\n\n----------------------------------------------------------------\n\n")



class TestValueDict:
    def __init__(self):

        self.set_value_lists()
        self.location_in_lists = {'gaussian': 0, 'rescale': 0, 'brighten': 0, 'contrast': 0, 'sharpen': 0,
                                  'averaging': 0, 'median': 0, 'closing': 0, 'erosion': 0, 'dilation': 0}

    def set_value_lists(self):
        self.value_dict = {'gaussian': [7, 1,3,5,7,9],
                            'rescale': [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],
                            'brighten': [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],
                           'contrast': [2, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],
                            'sharpen': [2, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],

                           'averaging': [1, 3, 5, 7, 9],
                           'median': [1, 3, 5, 7, 9],
                           'closing': [1, 3, 5, 7, 9],
                           'erosion': [1,3,5,7,9],
                           'dilation': [1,3,5,7,9]
                       }

    # 'rescale': [0.5, 1, 1.5, 2, 2.5, 3],
    # 'brighten': [0.5, 1, 1.5, 2, 2.5, 3],
    # 'contrast': [0.5, 1, 1.5, 2, 2.5, 3],
    # 'sharpen': [0.5, 1, 1.5, 2, 2.5, 3],


        quick_for_testing = { 'gaussian': [7],
                              'rescale': [1],
                              'brighten': [0.5],
                              'contrast': [2],
                              'sharpen': [2]}

        actual_big_test = {'gaussian': [1, 3, 5, 7],
                           'rescale': [0.5, 1, 1.5, 2, 2.5, 3],
                           'brighten': [0.5, 1, 1.5, 2, 2.5, 3],
                           'contrast': [0.5, 1, 1.5, 2, 2.5, 3],
                           'sharpen': [0.5, 1, 1.5, 2, 2.5, 3]}

        # self.value_dict = quick_for_testing


    def get_value_list(self, value_type):
        return self.value_dict[value_type]

    def inc_location_in_list(self, value_type):
        self.location_in_lists[value_type] = self.location_in_lists[value_type] + 1

    def reset_location_in_list(self, value_type):
        self.location_in_lists[value_type] = 0

    def get_value(self, value_type):
        location = self.location_in_lists[value_type]
        return self.value_dict[value_type][location]

