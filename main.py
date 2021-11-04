import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import os
import time
import preprocessing as pre
from datetime import datetime
import accuracy_testing as accuracy

def get_text(file):

    txt_file = create_output_file(file)
    path = 'pics/' + file
    delete_after = False

    if file.endswith('.pdf'):
        delete_after = True
        path = convert_pdf(path)

    img = cv2.imread(path)
    img = pre.deskew(img)
    img = pre.rescale(img, 2, 2)
    #img = pre.canny(img)
    #img = pre.erode(img)
    #img = pre.dilate(img)


    converted = pytesseract.image_to_string(img)
    print("Text found in '{}'\n".format(file))
    print(converted)

    add_to_file(txt_file, converted, pre.method_list)
    print(accuracy.string_similarity(converted, accuracy.actual))
    print(accuracy.levenshtein_similarity(converted, accuracy.actual))


    if delete_after:
        os.remove(path)  # get rid of the extra image file once we're done with it, if we made one


def convert_pdf(path):
    new_name = get_new_name(path)
    image = convert_from_path(path, paths_only=True, output_folder='converted_jpgs', fmt='jpeg', output_file=new_name)
    return image[0]


def get_new_name(path):
    split_path = path.split('/')
    pdf_name = split_path[len(split_path)-1]
    new_name = pdf_name.rstrip('.pdf')
    return new_name

def create_output_file(img_name):
    txt_name = img_name.split('.')[0]
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt = now.strftime("_%d_%m_%Y_%H:%M:%S")
    file = open('converted_text/' + txt_name + dt + '.txt', "w+")
    return file


def add_to_file(file, body_text=0, processing_methods=0):
    if body_text:
        # write the body
        file.write(body_text)
        file.write("\n\n----------------------------------TEXT COMPLETE----------------------------------------\n\n")
    if processing_methods:
        # write the processing methods
        file.write("The pre-processing methods used in this OCR text conversion were:\n")
        for method in processing_methods:
            file.write(method + '\n')
    if not body_text and not processing_methods:
        print("add_to_file function was called but not passed anything to add. /n")

    return file

if __name__ == '__main__':

    start_time = time.time()

    #get_text('Multiple-X-Ray.pdf')
    #get_text('learn_header.jpg')
    get_text('ocr_test_image.pdf')

    executionTime = (time.time() - start_time)
    print("Execution Time: {} seconds".format(round(executionTime, 2)))